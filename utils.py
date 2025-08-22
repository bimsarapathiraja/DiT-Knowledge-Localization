# Import required libraries for utility functions
import torch                            # PyTorch tensor operations
from transformers import PreTrainedTokenizer  # HuggingFace tokenizer interface


def find_substring_token_indices(prompt: str, substr: str, tokenizer: PreTrainedTokenizer, model="pixart"):
    """
    Find token indices corresponding to a substring within a tokenized prompt.
    
    This function locates the token positions of a target knowledge substring within
    a larger prompt, accounting for model-specific tokenization differences.
    
    Args:
        prompt: Full text prompt containing the target substring
        substr: Target substring to locate (e.g., artist name, place name)
        tokenizer: Model-specific tokenizer for text processing
        model: Model type ("pixart", "sana", or "flux") for tokenization handling
        
    Returns:
        list: Indices of tokens corresponding to the substring in the tokenized prompt
        
    Raises:
        AssertionError: If substring tokens are not found in the prompt tokens
    """
    # Ensure the model type is supported
    assert model in ["pixart", "sana", "flux"], f"Model {model} not supported"

    # Tokenize the full prompt to get token IDs
    prompt_tokens = tokenizer(prompt).input_ids

    # Handle model-specific tokenization differences
    if model == "pixart":
        # PixArt: Remove the last token (likely EOS token) from substring tokenization
        substr_tokens = tokenizer(substr).input_ids[:-1]
    elif model == "sana":
        # Sana: Handle special case where "in the style of" requires space prefix
        if "in the style of" in prompt: # TODO: Clean code [for sana, " X" and "X" resul in different tokens]
            substr = " " + substr  # Add space prefix for consistent tokenization
        else:
            substr = substr        # Use substring as-is
        # Remove the first token (likely BOS token) from substring tokenization
        substr_tokens = tokenizer(substr).input_ids[1:]
    elif model == "flux":
        # Flux: Remove both first and last tokens (BOS and EOS tokens) from substring
        substr_tokens = tokenizer(substr).input_ids[1:-1]
    else:
        # This should not be reached due to assert above, but included for completeness
        raise ValueError(f"Model {model} not recognized.")

    # Search for substring tokens within the prompt tokens using sliding window
    start_idx = -1
    for i in range(len(prompt_tokens) - len(substr_tokens) + 1):
        # Check if tokens at position i match the substring tokens exactly
        if prompt_tokens[i:i+len(substr_tokens)] == substr_tokens:
            start_idx = i  # Found the starting position
            break   

    # Ensure the substring was found in the prompt
    assert start_idx != -1, "substr_tokens not found in tokens"

    # Generate list of consecutive token indices for the substring
    token_indices = list(range(start_idx, start_idx + len(substr_tokens)))

    # Verify tokenization correctness by decoding and comparing
    if tokenizer.decode([prompt_tokens[token_idx] for token_idx in token_indices]) != substr:
        # Print warning if decoded tokens don't exactly match original substring
        print("============================ Warning ============================")
        print("[Warning] tokenizer.decode([prompt_tokens[token_idx] for token_idx in token_indices]) != substr")
        print(f"[Warning] Decoded: {tokenizer.decode([prompt_tokens[token_idx] for token_idx in token_indices])}")
        print(f"[Warning] Expected: {substr}")
        print("=================================================================")

    return token_indices


def latents_to_images(pipe, latents):
    """
    Convert latent representations to PIL images using the pipeline's VAE decoder.
    
    This function decodes latent representations back to pixel space images,
    applying proper scaling and post-processing for visualization.
    
    Args:
        pipe: Diffusion pipeline containing VAE decoder and image processor
        latents: Latent tensor representations [batch, channels, height, width]
        
    Returns:
        list: PIL Images converted from latent representations
    """
    # Convert latents to VAE's expected data type for compatibility
    latents = latents.to(pipe.vae.dtype)
    
    # Decode latents to pixel space without computing gradients
    with torch.no_grad():
        # Apply VAE scaling factor to denormalize latents before decoding
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    
    # Post-process decoded images to PIL format with proper scaling/clamping
    images = pipe.image_processor.postprocess(images, output_type="pil")
    
    return images


def get_worker_list_chunk(arr, num_workers, worker_idx, print_log=True):
    """
    Split a list into chunks for distributed processing across multiple workers.
    
    This function divides work among parallel workers by assigning each worker
    a contiguous subset of the input array. Useful for parallel knowledge localization.
    
    Args:
        arr: List or array to split among workers
        num_workers: Total number of parallel workers
        worker_idx: Current worker's index (0 to num_workers-1)
        print_log: Whether to print chunk information for debugging
        
    Returns:
        list: Subset of the array assigned to this worker
    """
    # Get total length of the input array
    arr_len = len(arr)

    # Calculate chunk size, ensuring all elements are covered with ceiling division
    chunk_size = (arr_len + num_workers - 1) // num_workers

    # Calculate start and end indices for this worker's chunk
    start_index = chunk_size * worker_idx                    # Starting position for this worker
    end_index = min((worker_idx + 1) * chunk_size, arr_len) # Ending position (clamped to array length)

    # Print chunk information for debugging and monitoring
    if print_log:
        print(f"Choosing chunk ({start_index}:{end_index})")
        print(f"First prompt of the chunk: \"{arr[start_index]}\"")
        print(f"Last prompt of the chunk: \"{arr[end_index-1]}\"")

    # Return the slice assigned to this worker
    return arr[start_index:end_index]


def print_arguments(args):
    """
    Print all parsed command-line arguments in a formatted way.
    
    This utility function displays all configuration parameters for debugging
    and logging purposes, making it easy to verify experiment settings.
    
    Args:
        args: Namespace object containing parsed command-line arguments
    """
    print("===================== Arguments =====================")
    # Iterate through all argument attributes and their values
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=====================================================")
