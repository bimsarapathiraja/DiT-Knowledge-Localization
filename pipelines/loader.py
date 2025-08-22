# Import PyTorch for model operations
import torch


def load_pipe(model_name):
    """
    Load and configure a diffusion pipeline based on the specified model name.
    
    This function initializes different diffusion models with appropriate configurations
    for knowledge localization experiments. Each model requires specific setup.
    
    Args:
        model_name: String identifier for the model ("pixart", "sana", or "flux")
        
    Returns:
        Pipeline: Configured diffusion pipeline ready for knowledge localization
        
    Raises:
        ValueError: If the model name is not recognized
    """
    if model_name == "pixart":
        # Load PixArt-Alpha pipeline for high-quality text-to-image generation
        from pipelines.pixart_alpha import CustomPixArtAlphaPipeline

        # Initialize PixArt pipeline with FP16 precision for memory efficiency
        pipe = CustomPixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",  # Pre-trained PixArt model checkpoint
            torch_dtype=torch.float16,            # Use half precision to reduce VRAM usage
            local_files_only=True,                # Use locally cached model files
        ).to("cuda")  # Move pipeline to GPU for acceleration
        
        # Set up custom attention processors for localization experiments
        pipe.set_custom_attn_processor()
        
    elif model_name == "sana":
        # Load Sana pipeline for efficient high-resolution image generation
        from pipelines.sana import CustomSanaPipeline
        
        # Initialize Sana pipeline with full precision for stability
        pipe = CustomSanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",  # Pre-trained Sana model
            torch_dtype=torch.float32  # Use full precision for better numerical stability
        ).to("cuda")  # Move pipeline to GPU
        
        # Convert specific components to bfloat16 for optimal performance
        pipe.text_encoder.to(torch.bfloat16)  # Text encoder can use lower precision
        pipe.transformer = pipe.transformer.to(torch.bfloat16)  # Transformer model precision
        
    elif model_name == "flux":
        # Load Flux pipeline for state-of-the-art text-to-image generation
        from pipelines.flux import CustomFluxPipeline
        
        # Initialize Flux pipeline with bfloat16 for optimal memory/quality balance
        pipe = CustomFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",  # Official Flux development model
            torch_dtype=torch.bfloat16       # Use bfloat16 for better stability than fp16
        ).to("cuda")  # Move pipeline to GPU for fast inference
        
    else:
        # Raise error for unsupported model names
        raise ValueError(f"Model {model_name} not recognized.")

    return pipe
