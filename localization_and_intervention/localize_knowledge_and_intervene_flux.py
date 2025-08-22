# Import required standard library modules
import argparse  # For parsing command-line arguments
import json      # For JSON file operations (saving results)
import os        # For operating system interface operations (file/directory management)
import sys       # For system-specific parameters and functions
from functools import partial  # For creating partial function objects
from pathlib import Path       # For object-oriented filesystem paths
import inspect   # For inspecting live objects (unused in current version)

# Import PyTorch and related libraries
import torch                                    # PyTorch deep learning framework
from tqdm import tqdm                           # Progress bar library for loops
from transformers import CLIPProcessor, CLIPModel  # HuggingFace CLIP model and processor

# Add parent directory to Python path to access project modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import project-specific utility functions
from utils import find_substring_token_indices, get_worker_list_chunk, print_arguments
# Import custom attention processors for Flux model
from attention_processor import FluxAttnContCalculatorProcessor, FluxEmbeddingModifierAttnProcessor
# Import CLIP scoring functionality
from clip_score import get_clip_score
# Import dataset utilities for knowledge localization
from dataset import get_knowledge_dataset_class_and_get_list_fn, get_eval_text_for_knowledge
# Import pipeline loader for loading Flux model
from pipelines.loader import load_pipe


def localize_dominant_blocks(args, pipe, dataset):
    """
    Localize the most influential transformer blocks for a specific knowledge type.
    
    This function identifies which transformer blocks contribute most to generating
    specific knowledge by measuring attention contributions for relevant tokens.
    
    Args:
        args: Command line arguments containing configuration
        pipe: Loaded Flux pipeline with transformer model
        dataset: Dataset containing prompts with the target knowledge
    
    Returns:
        tuple: (mm_dominant_blocks, single_dominant_blocks) - lists of block indices
    """
    # Set up attention contribution calculators for all transformer blocks
    # Both multi-modal (MM) and single transformer blocks are processed
    for idx, basic_transformer_block in enumerate(pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks):
        # Replace default attention processor with contribution calculator
        # The -1 parameter indicates token indices will be set later per prompt
        basic_transformer_block.attn.set_processor(FluxAttnContCalculatorProcessor(-1))

    # Initialize tensor to accumulate attention contributions for all blocks
    # Length equals total number of transformer blocks (MM + single)
    aggergated_attn_cont = torch.zeros(len(pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks))

    # Process each prompt in the dataset to measure attention contributions
    for prompt in tqdm(dataset, desc="Localizing Dominant Blocks"):
        # Find token indices in the prompt that correspond to the target knowledge
        # These are the tokens we want to measure attention contributions for
        token_indices = find_substring_token_indices(prompt, dataset.knowledge, pipe.tokenizer, "flux")
        
        # Set the target token indices for all attention processors
        # This tells each processor which tokens to focus on when calculating contributions
        for block in pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks:
            block.attn.processor.token_indices_for_attn_cont_calc = token_indices

        # Generate image from prompt without gradients (inference only)
        # During this forward pass, attention processors accumulate contribution measurements
        with torch.no_grad():
            _ = pipe(
                prompt=prompt,                              # Input text prompt
                output_type="latent",                       # Return latent representation (faster)
                generator=torch.Generator().manual_seed(args.generator_seed),  # Fixed seed for reproducibility
            )
    
    # Collect accumulated attention contributions from each block's processor
    for idx, block in enumerate(pipe.transformer.transformer_blocks + pipe.transformer.single_transformer_blocks):
        # Average the contribution over all update counts to get mean contribution per block
        aggergated_attn_cont[idx] = block.attn.processor.attn_contribution / block.attn.processor.attn_contribution_update_count

    # Save the attention contributions to disk for analysis
    torch.save(aggergated_attn_cont, f"{args.results_path}/{dataset.knowledge}/attn_cont.pt")

    # Find the top-k blocks with highest attention contributions
    # These are the blocks most involved in generating the target knowledge
    top_dominant_blocks = aggergated_attn_cont.topk(args.disable_k_dominant_blocks).indices

    # Separate dominant blocks into multi-modal (MM) and single transformer categories
    mm_attn_embedding_modifier_indices = []      # Indices for MM transformer blocks
    single_attn_embedding_modifier_indices = []  # Indices for single transformer blocks

    # Categorize each dominant block based on its position in the architecture
    for idx in top_dominant_blocks.tolist():
        if idx < len(pipe.transformer.transformer_blocks):
            # Block belongs to multi-modal transformer blocks
            mm_attn_embedding_modifier_indices.append(idx)
        else:
            # Block belongs to single transformer blocks (adjust index accordingly)
            single_attn_embedding_modifier_indices.append(idx - len(pipe.transformer.transformer_blocks))

    return mm_attn_embedding_modifier_indices, single_attn_embedding_modifier_indices


def evaluate(args, pipe, clip_score_fn, dataset, mm_top_dominant_blocks, single_top_dominant_blocks):
    """
    Evaluate knowledge intervention by generating images with modified attention.
    
    This function performs intervention by replacing encoder hidden states in the 
    identified dominant blocks, effectively "removing" or "replacing" the target knowledge.
    
    Args:
        args: Command line arguments with evaluation settings
        pipe: Flux pipeline for image generation
        clip_score_fn: Function to compute CLIP similarity scores
        dataset: Dataset containing test prompts 
        mm_top_dominant_blocks: Multi-modal transformer block indices to intervene
        single_top_dominant_blocks: Single transformer block indices to intervene
    """
    # Set up embedding modifier processors for all multi-modal transformer blocks
    # These processors will handle the intervention during generation
    for idx, block in enumerate(pipe.transformer.transformer_blocks):
        # Each block gets a FluxEmbeddingModifierAttnProcessor to handle embedding replacement
        block.attn.set_processor(FluxEmbeddingModifierAttnProcessor())
    
    # Dictionary to store evaluation results (filename -> CLIP score)
    res = {}
    
    # Generate and evaluate images for each prompt in the dataset
    for prompt in tqdm(dataset, desc="Evaluating"):
        # Generate multiple images per prompt for robust evaluation
        for i in range(args.num_images_per_eval_prompt):
            # Create unique seed for each image to ensure variety
            seed = args.generator_seed + i

            # Generate image with knowledge intervention
            with torch.no_grad():
                image = pipe(
                    prompt=prompt,                                                    # Original prompt containing target knowledge
                    clean_prompt=dataset.get_clean_prompt(prompt),                   # Modified prompt without target knowledge
                    mm_attn_embedding_modifier_indices=mm_top_dominant_blocks,       # MM blocks to modify
                    single_attn_embedding_modifier_indices=single_top_dominant_blocks, # Single blocks to modify
                    replace_pooled_prompt_embeds=True,                               # Use clean prompt's pooled embeddings
                    generator=torch.Generator().manual_seed(seed),                   # Fixed seed for reproducibility
                ).images[0].resize((512, 512))  # Resize to standard size for consistent evaluation

            # Create unique filename for this generated image
            file_name = f"{prompt}_{seed}.png"

            # Save the generated image to results directory
            image.save(f"{args.results_path}/{dataset.knowledge}/{file_name}")
            
            # Compute CLIP similarity score between the image and evaluation text
            # This measures how well the intervention worked (lower score = better intervention)
            res[file_name] = clip_score_fn(get_eval_text_for_knowledge(args.knowledge_type, dataset.knowledge), image)
    
    # Save all evaluation results to JSON file
    with open(f"{args.results_path}/{dataset.knowledge}/results.json", "w") as f:
        json.dump(res, f, indent=4)


def parse_args():
    """
    Parse and validate command line arguments for knowledge localization.
    
    Returns:
        Namespace: Parsed arguments with all configuration parameters
    """
    # Create argument parser for command line interface
    parser = argparse.ArgumentParser()
    
    # Number of timesteps for diffusion process (default for Flux)
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=28,  # Standard timesteps for Flux model generation
    )
    
    # Random seed for reproducible generation
    parser.add_argument(
        "--generator_seed",
        type=int,
        default=0,   # Deterministic seed for consistent results
    )
    
    # Number of most dominant blocks to identify and intervene on
    parser.add_argument(
        "--disable_k_dominant_blocks",
        type=int,
        default=6,   # Top-6 blocks typically capture core knowledge
    )
    
    # Path where results will be saved (required parameter)
    parser.add_argument(
        "--results_path",
        type=str,
        required=True  # Must be provided by user
    )
    
    # Number of images to generate per evaluation prompt
    parser.add_argument(
        "--num_images_per_eval_prompt",
        type=int,
        default=3,   # Multiple images for robust evaluation statistics
    )
    
    # Total number of parallel workers for distributed processing
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,  # Parallel processing for faster completion
    )
    
    # Current worker index (0 to num_workers-1) for distributed processing
    parser.add_argument(
        "--worker_idx",
        type=int,
        default=None,
        required=True,  # Each worker must know its index
    )
    
    # Attention contribution calculation variant (currently only one option)
    parser.add_argument(
        "--calc_attn_cont_fn_variant",
        type=str,
        default="m_mult_v_mult_o",  # Matrix multiplication with value and output
        choices=["m_mult_v_mult_o"], # Only this variant is implemented for Flux
    )
    
    # Type of knowledge to localize (must be one of supported categories)
    parser.add_argument(
        "--knowledge_type",
        type=str,
        required=True,
        default=None,
        choices=["style", "place", "copyright", "animal", "celebrity", "safety"]  # Supported knowledge categories
    )

    # Parse all provided arguments
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Parse command line arguments to get configuration
    args = parse_args()
    
    # Display all parsed arguments for debugging and logging
    print_arguments(args)

    # Get the appropriate dataset class and knowledge list function for the specified knowledge type
    # The dataset class provides prompts and clean prompts for the knowledge category
    # The get_knowledge_list_fn returns all available knowledge items in that category (e.g., all artists for "style")
    dataset_class, get_knowledge_list_fn = get_knowledge_dataset_class_and_get_list_fn(args.knowledge_type, for_model="flux")

    # Distribute the knowledge list among multiple workers for parallel processing
    # Each worker processes a subset of the total knowledge items
    worker_knowledge_list = get_worker_list_chunk(get_knowledge_list_fn(), args.num_workers, args.worker_idx)

    # Create results directory if it doesn't exist
    if not os.path.exists(args.results_path):
        print(f"Creating path for {args.results_path}")
        os.makedirs(args.results_path, exist_ok=True)  # Create directory and any necessary parent directories

    # Load the Flux model pipeline with all necessary components
    # This includes the transformer, VAE, text encoder, and scheduler
    pipe = load_pipe("flux")

    # Load CLIP model and processor for evaluation metrics
    # CLIP measures semantic similarity between generated images and text descriptions
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

    # Process each knowledge item assigned to this worker
    for knowledge in worker_knowledge_list:
        print(f"Start Localizing for {args.knowledge_type.title()}: {knowledge}")

        # Skip processing if results already exist for this knowledge item
        if os.path.exists(f"{args.results_path}/{knowledge}/results.json"):
            print(f"Results already exists for {knowledge}. Skipping...")
            continue
        
        # Create subdirectory for this specific knowledge item's results
        if not os.path.exists(f"{args.results_path}/{knowledge}"):
            os.makedirs(f"{args.results_path}/{knowledge}")

        # Step 1: Localize the dominant transformer blocks for this knowledge
        # Uses training prompts to identify blocks most responsible for generating this knowledge
        mm_top_dominant_blocks, single_top_dominant_blocks = localize_dominant_blocks(args, pipe, dataset_class(knowledge, "train"))

        # Log the identified dominant blocks for analysis
        print(f"Most MM dominant blocks for {args.knowledge_type} \"{knowledge}\": {mm_top_dominant_blocks}")
        print(f"Most single dominant blocks for {args.knowledge_type} \"{knowledge}\": {single_top_dominant_blocks}")

        # Step 2: Evaluate intervention effectiveness using the identified dominant blocks
        # Uses both training and test prompts to generate images with knowledge intervention
        print(f"Start Evaluation for {args.knowledge_type.title()}: {knowledge}")
        evaluate(args, pipe, partial(get_clip_score, clip_model, clip_processor), dataset_class(knowledge, "both"), mm_top_dominant_blocks, single_top_dominant_blocks)
