# Import required modules for custom Flux pipeline implementation
import inspect                      # For function signature inspection
from typing import Callable, List, Optional, Dict, Union, Any  # Type hints for better code documentation

import numpy as np                  # Numerical operations for timestep scheduling
import torch                        # PyTorch tensor operations

# Import Diffusers components for Flux pipeline
from diffusers import FluxPipeline                    # Base Flux pipeline class
from diffusers.pipelines.flux import FluxPipelineOutput  # Standard output format
from diffusers.utils import is_torch_xla_available   # Check for TPU availability
from diffusers.image_processor import PipelineImageInput  # Input image type hint

# Add parent directory to path for importing custom attention processors
import sys
sys.path.append("..")
from attention_processor import FluxEmbeddingModifierAttnProcessor  # Custom attention processor for intervention

# Handle optional TPU support
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm  # TPU operations
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    """
    Calculate timestep shift parameter for Flux model based on image sequence length.
    
    Flux uses a timestep shift mechanism to adapt the diffusion schedule based on
    the complexity of the image (measured by sequence length). Longer sequences
    require different noise schedules for optimal generation quality.
    
    Args:
        image_seq_len: Length of the image token sequence after patchification
        base_seq_len: Minimum sequence length for shift calculation (default: 256)
        max_seq_len: Maximum sequence length for shift calculation (default: 4096)  
        base_shift: Shift value for minimum sequence length (default: 0.5)
        max_shift: Shift value for maximum sequence length (default: 1.16)
        
    Returns:
        float: Calculated shift parameter for timestep scheduling
    """
    # Linear interpolation between base_shift and max_shift based on sequence length
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)  # Slope of linear function
    b = base_shift - m * base_seq_len                           # Y-intercept
    mu = image_seq_len * m + b                                  # Final shift value
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Retrieve and configure timesteps for the diffusion process.
    
    This function handles different ways of specifying the diffusion schedule:
    either through number of steps, explicit timesteps, or sigma values.
    It validates the scheduler compatibility and sets up the timestep schedule.
    
    Args:
        scheduler: The diffusion scheduler to configure
        num_inference_steps: Number of denoising steps to perform
        device: Device to place timesteps tensor on
        timesteps: Explicit list of timestep values (overrides num_inference_steps)
        sigmas: Explicit list of noise sigma values (overrides other parameters)
        **kwargs: Additional arguments passed to scheduler.set_timesteps()
        
    Returns:
        tuple: (timesteps tensor, actual number of inference steps)
        
    Raises:
        ValueError: If both timesteps and sigmas are provided, or scheduler doesn't support them
    """
    # Validate that only one timestep specification method is used
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    
    if timesteps is not None:
        # Use explicit timesteps if provided
        # Check if the scheduler supports custom timestep schedules
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        # Configure scheduler with custom timesteps
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps          # Get the configured timesteps
        num_inference_steps = len(timesteps)     # Update step count to match timesteps
        
    elif sigmas is not None:
        # Use explicit sigma values if provided
        # Check if the scheduler supports custom sigma schedules  
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        # Configure scheduler with custom sigmas
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps          # Get timesteps corresponding to sigmas
        num_inference_steps = len(timesteps)     # Update step count
        
    else:
        # Use standard number of inference steps (most common case)
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps          # Get the standard timestep schedule
        
    return timesteps, num_inference_steps


class CustomFluxPipeline(FluxPipeline):
    """
    Extended Flux pipeline with knowledge intervention capabilities.
    
    This custom pipeline extends the standard FluxPipeline to support knowledge localization
    and intervention experiments. It adds the ability to:
    1. Use "clean" embeddings (without target knowledge) during generation
    2. Selectively modify attention in specific transformer blocks
    3. Generate both original and intervention images for comparison
    """
    
    def set_clean_scheduler(self):
        """
        Create a copy of the main scheduler for processing "clean" embeddings.
        
        During knowledge intervention, we need to run two parallel diffusion processes:
        one with original embeddings and one with clean embeddings. This method
        creates an independent scheduler copy for the clean path.
        """
        # Clone the main scheduler configuration to create an independent copy
        self.clean_scheduler = self.scheduler.__class__.from_config(self.scheduler.config)
    
    def set_embedding_modifier_attn_processor(self):
        """
        Install embedding modifier attention processors on all transformer blocks.
        
        This method replaces the standard attention processors with custom ones that
        can save and replace encoder hidden states for knowledge intervention.
        """
        # Install custom processors on multi-modal transformer blocks
        for block in self.transformer.transformer_blocks:
            block.attn.set_processor(FluxEmbeddingModifierAttnProcessor())
    
        # Install custom processors on single transformer blocks  
        for block in self.transformer.single_transformer_blocks:
            block.attn.set_processor(FluxEmbeddingModifierAttnProcessor())

    def change_attn_embedding_modifier_mode(self, mm_attn_embedding_modifier_indices, single_attn_embedding_modifier_indices, mode):
        """
        Change the processing mode for specific attention processors during intervention.
        
        This method switches between saving clean embeddings and using them for replacement
        in the identified dominant blocks.
        
        Args:
            mm_attn_embedding_modifier_indices: Multi-modal block indices to modify
            single_attn_embedding_modifier_indices: Single transformer block indices to modify  
            mode: ProcessorMode (SAVE_ENCODER_HIDDEN_STATES or REPLACE_ENCODER_HIDDEN_STATES)
        """
        # Set mode for specified multi-modal transformer blocks
        for idx in mm_attn_embedding_modifier_indices:
            self.transformer.transformer_blocks[idx].attn.processor.mode = mode

        # Set mode for specified single transformer blocks
        for idx in single_attn_embedding_modifier_indices:
            self.transformer.single_transformer_blocks[idx].attn.processor.mode = mode
    
    def clear_attn_embedding_modifier_processors_cache(self):
        """
        Clear cached encoder hidden states from all embedding modifier processors.
        
        This frees memory after knowledge intervention is complete by removing
        stored clean embeddings from all attention processors.
        """
        # Clear cache from multi-modal transformer blocks
        for block in self.transformer.transformer_blocks:
            block.attn.processor.clear_cache()
        
        # Clear cache from single transformer blocks
        for block in self.transformer.single_transformer_blocks:
            block.attn.processor.clear_cache()
        
    @torch.no_grad()
    def __call__(
        self,
        # Standard Flux pipeline parameters
        prompt: Union[str, List[str]] = None,                              # Text prompt(s) for image generation  
        prompt_2: Optional[Union[str, List[str]]] = None,                 # Secondary prompts (unused in current implementation)
        negative_prompt: Union[str, List[str]] = None,                     # Negative prompt(s) to avoid certain content
        negative_prompt_2: Optional[Union[str, List[str]]] = None,        # Secondary negative prompts
        true_cfg_scale: float = 1.0,                                      # True classifier-free guidance scale
        height: Optional[int] = 512,                                      # Generated image height (changed from default)
        width: Optional[int] = 512,                                       # Generated image width (changed from default)
        num_inference_steps: int = 28,                                    # Number of diffusion denoising steps
        sigmas: Optional[List[float]] = None,                             # Custom noise schedule (overrides num_inference_steps)
        guidance_scale: float = 3.5,                                      # Classifier-free guidance strength
        num_images_per_prompt: Optional[int] = 1,                         # Number of images to generate per prompt
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  # Random number generator for reproducibility
        latents: Optional[torch.FloatTensor] = None,                      # Pre-computed initial noise (optional)
        prompt_embeds: Optional[torch.FloatTensor] = None,                # Pre-computed prompt embeddings (optional)
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,         # Pre-computed pooled prompt embeddings (optional)
        ip_adapter_image: Optional[PipelineImageInput] = None,            # Input image for IP-Adapter conditioning
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,     # Pre-computed IP-Adapter image embeddings
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,   # Negative IP-Adapter conditioning image
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,  # Negative IP-Adapter embeddings
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,       # Pre-computed negative prompt embeddings
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,  # Pre-computed negative pooled embeddings
        output_type: Optional[str] = "pil",                               # Output format ("pil", "latent", etc.)
        return_dict: bool = True,                                         # Whether to return FluxPipelineOutput object
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,          # Additional attention parameters
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,  # Callback function after each step
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],      # Tensors to pass to callback
        max_sequence_length: int = 512,                                   # Maximum text sequence length
        
        # Custom parameters for knowledge intervention
        clean_prompt: Union[str, List[str]] = None,                       # "Clean" prompt without target knowledge
        mm_attn_embedding_modifier_indices: Optional[List[int]] = None,   # Multi-modal blocks to modify for intervention
        single_attn_embedding_modifier_indices: Optional[List[int]] = None,  # Single blocks to modify for intervention
        replace_pooled_prompt_embeds: bool = False,                       # Use clean prompt's pooled embeddings
        return_clean_image: bool = False,                                 # Return both original and clean images
    ):
        """
        Generate images with optional knowledge intervention capabilities.
        
        This extended __call__ method supports standard Flux generation plus knowledge
        intervention. It can generate images with specific knowledge removed or modified
        by replacing encoder hidden states in identified dominant transformer blocks.
        
        The intervention process works in two phases:
        1. "Clean" pass: Generate with clean prompt and save encoder states
        2. "Intervention" pass: Generate with original prompt but replace encoder states
           in dominant blocks with clean states
        
        Returns:
            Union[FluxPipelineOutput, Tuple]: Generated images, optionally with clean comparison
        """
        # Set default image dimensions if not provided
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # Step 1: Validate all input parameters to ensure they're compatible
        self.check_inputs(
            prompt,                                    # Primary prompt validation
            prompt_2,                                  # Secondary prompt validation  
            height,                                    # Image height validation
            width,                                     # Image width validation
            negative_prompt=negative_prompt,           # Negative prompt validation
            negative_prompt_2=negative_prompt_2,       # Secondary negative prompt validation
            prompt_embeds=prompt_embeds,               # Pre-computed embeddings validation
            negative_prompt_embeds=negative_prompt_embeds,  # Negative embeddings validation
            pooled_prompt_embeds=pooled_prompt_embeds,      # Pooled embeddings validation
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,  # Negative pooled validation
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,  # Callback inputs validation
            max_sequence_length=max_sequence_length,   # Text sequence length validation
        )

        # Validate intervention parameters - both must be provided together or not at all
        assert (mm_attn_embedding_modifier_indices is None) == (single_attn_embedding_modifier_indices is None), \
            "If you want to use mm and single attn embedding modifier indices, please provide both or none"
        
        # Validate intervention indices are within valid ranges
        if mm_attn_embedding_modifier_indices is not None:
            assert len(mm_attn_embedding_modifier_indices) >= 0 and \
                   all([0 <= i < len(self.transformer.transformer_blocks) for i in mm_attn_embedding_modifier_indices]), \
                   "Multi-modal attention embedding modifier indices must be within transformer_blocks range"
            assert len(single_attn_embedding_modifier_indices) >= 0 and \
                   all([0 <= i < len(self.transformer.single_transformer_blocks) for i in single_attn_embedding_modifier_indices]), \
                   "Single attention embedding modifier indices must be within single_transformer_blocks range"
            assert clean_prompt is not None, "If you want to use clean pass, please provide a clean prompt"

        # Store pipeline configuration parameters
        self._guidance_scale = guidance_scale              # Classifier-free guidance strength
        self._joint_attention_kwargs = joint_attention_kwargs  # Additional attention parameters
        self._interrupt = False                           # Flag for interrupting generation mid-process

        # Step 2: Determine batch size from prompt or pre-computed embeddings
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1                               # Single string prompt
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)                     # List of prompts
        else:
            batch_size = prompt_embeds.shape[0]          # Use embedding batch size

        # Set device for tensor operations (typically CUDA for GPU acceleration)
        device = self._execution_device

        # Extract LoRA (Low-Rank Adaptation) scale if using fine-tuned models
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        
        # Determine if true classifier-free guidance is needed (requires negative prompts)
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        
        # Step 3: Encode the main prompt into embeddings
        # This converts text into the tensor representations used by the transformer
        (
            prompt_embeds,           # Text token embeddings [batch, seq_len, dim]
            pooled_prompt_embeds,    # Pooled/aggregated text representation [batch, dim]
            text_ids,                # Token position IDs for attention
        ) = self.encode_prompt(
            prompt=prompt,                              # Input text prompt
            prompt_2=prompt_2,                          # Secondary prompt (unused)
            prompt_embeds=prompt_embeds,                # Use pre-computed embeddings if provided
            pooled_prompt_embeds=pooled_prompt_embeds,  # Use pre-computed pooled embeddings if provided
            device=device,                              # Target device for embeddings
            num_images_per_prompt=num_images_per_prompt,  # Repeat embeddings for multiple images
            max_sequence_length=max_sequence_length,    # Maximum text length
            lora_scale=lora_scale,                      # LoRA scaling factor
        )
        
        # Encode negative prompt if using classifier-free guidance
        if do_true_cfg:
            (
                negative_prompt_embeds,        # Negative text embeddings
                negative_pooled_prompt_embeds, # Negative pooled embeddings  
                _,                             # Text IDs (same as positive, discarded)
            ) = self.encode_prompt(
                prompt=negative_prompt,                          # Negative prompt text
                prompt_2=negative_prompt_2,                      # Secondary negative prompt
                prompt_embeds=negative_prompt_embeds,            # Pre-computed negative embeddings
                pooled_prompt_embeds=negative_pooled_prompt_embeds,  # Pre-computed negative pooled
                device=device,                                   # Target device
                num_images_per_prompt=num_images_per_prompt,     # Batch replication
                max_sequence_length=max_sequence_length,         # Text length limit
                lora_scale=lora_scale,                          # LoRA scaling
            )
        
        # Step 4: Set up knowledge intervention if parameters are provided
        do_clean_pass = mm_attn_embedding_modifier_indices is not None  # Enable intervention mode
        
        # Validate clean image return requirements
        if return_clean_image:
            assert do_clean_pass, "If you want to return clean image, please provide a clean prompt and mm_attn_embedding_modifier_indices"
        
        # Configure intervention system if needed
        if do_clean_pass:
            # Install custom attention processors that can save/replace encoder states
            self.set_embedding_modifier_attn_processor()
        
            # Encode the "clean" prompt (without target knowledge)
            (
                clean_prompt_embeds,        # Clean text embeddings without target knowledge
                clean_pooled_prompt_embeds, # Clean pooled embeddings
                text_ids,                   # Position IDs (reused)
            ) = self.encode_prompt(
                prompt=clean_prompt,                        # Clean prompt without target knowledge
                prompt_2=None,                              # No secondary clean prompt
                device=device,                              # Target device
                num_images_per_prompt=num_images_per_prompt,  # Batch size
                max_sequence_length=max_sequence_length,    # Text length limit
                lora_scale=lora_scale,                      # LoRA scaling
            )

            # Optionally use clean prompt's pooled embeddings for the main generation
            if replace_pooled_prompt_embeds:
                pooled_prompt_embeds = clean_pooled_prompt_embeds
            # Note: Could also replace prompt_embeds here, but currently commented out
            # prompt_embeds = clean_prompt_embeds
            
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if do_clean_pass:
            clean_latents, clean_latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        if do_clean_pass:
            self.set_clean_scheduler()
            self.clean_scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if do_clean_pass:
                    self.change_attn_embedding_modifier_mode(
                        mm_attn_embedding_modifier_indices,
                        single_attn_embedding_modifier_indices,
                        FluxEmbeddingModifierAttnProcessor.ProcessorMode.SAVE_ENCODER_HIDDEN_STATES
                    )

                    clean_noise_pred = self.transformer(
                        hidden_states=clean_latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=clean_pooled_prompt_embeds,
                        encoder_hidden_states=clean_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=clean_latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    self.change_attn_embedding_modifier_mode(
                        mm_attn_embedding_modifier_indices,
                        single_attn_embedding_modifier_indices,
                        FluxEmbeddingModifierAttnProcessor.ProcessorMode.REPLACE_ENCODER_HIDDEN_STATES
                    )

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    raise Exception("Not Handled Yet!")
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if do_clean_pass:
                    clean_latents = self.clean_scheduler.step(clean_noise_pred, t, clean_latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)
                        if do_clean_pass:
                            clean_latents = clean_latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()
        
        if do_clean_pass:
            self.change_attn_embedding_modifier_mode(
                mm_attn_embedding_modifier_indices,
                single_attn_embedding_modifier_indices,
                FluxEmbeddingModifierAttnProcessor.ProcessorMode.NONE
            )
            self.clear_attn_embedding_modifier_processors_cache()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

            if do_clean_pass and return_clean_image:
                clean_latents = self._unpack_latents(clean_latents, height, width, self.vae_scale_factor)
                clean_latents = (clean_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                clean_image = self.vae.decode(clean_latents, return_dict=False)[0]
                clean_image = self.image_processor.postprocess(clean_image, output_type=output_type)
                
                return image, clean_image

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

"""
=== KNOWLEDGE LOCALIZATION IN FLUX TRANSFORMERS: COMPLETE WORKFLOW ===

This file implements a comprehensive system for localizing and intervening on specific 
knowledge in Flux diffusion models. Here's how the complete workflow operates:

## 1. KNOWLEDGE LOCALIZATION PHASE (localize_knowledge_and_intervene_flux.py)

The localization process identifies which transformer blocks are most responsible for 
generating specific knowledge (e.g., artistic styles, places, objects):

### Step 1: Attention Contribution Measurement
- FluxAttnContCalculatorProcessor replaces standard attention processors
- For each training prompt containing target knowledge:
  * Token indices of target knowledge are identified (e.g., "Van Gogh" tokens)
  * During forward pass, attention contributions to these tokens are measured
  * Contributions quantify how much each block influences target token generation

### Step 2: Dominant Block Identification  
- Attention contributions are aggregated across all training prompts
- Top-k blocks with highest contributions are identified as "dominant blocks"
- These blocks are separated into multi-modal (MM) and single transformer categories

## 2. KNOWLEDGE INTERVENTION PHASE (CustomFluxPipeline.__call__)

The intervention process removes or modifies target knowledge by replacing encoder
hidden states in the identified dominant blocks:

### Step 1: Dual Forward Pass Setup
- Clean prompt: Original prompt with target knowledge removed (e.g., "A painting" vs "A Van Gogh painting")  
- FluxEmbeddingModifierAttnProcessor replaces standard processors in dominant blocks

### Step 2: Clean Pass (Save Phase)
- Generate with clean prompt to get "knowledge-free" representations
- Processors in SAVE_ENCODER_HIDDEN_STATES mode cache clean text embeddings
- Clean embeddings represent what the model should generate without target knowledge

### Step 3: Intervention Pass (Replace Phase) 
- Generate with original prompt but replace encoder states in dominant blocks
- Processors in REPLACE_ENCODER_HIDDEN_STATES mode substitute cached clean embeddings
- This "removes" target knowledge while preserving other semantic content

### Step 4: Evaluation
- Compare generated images with/without intervention using CLIP similarity scores
- Lower CLIP scores indicate successful knowledge removal
- Multiple images per prompt provide robust evaluation statistics

## 3. KEY TECHNICAL INNOVATIONS

### Attention Contribution Calculation
The calc_attn_cont() method computes knowledge influence as:
```
attention_to_knowledge * knowledge_values -> output_influence  
contribution = L2_norm(output_influence).mean()
```

### Selective Encoder State Replacement
Only dominant blocks get modified, preserving other knowledge and capabilities:
```
if block_idx in dominant_blocks:
    encoder_states = clean_encoder_states  # Remove target knowledge
else:
    encoder_states = original_encoder_states  # Preserve other knowledge
```

### Multi-Modal Architecture Handling
Flux has two types of transformer blocks:
- Multi-modal blocks: Handle both text and image tokens
- Single blocks: Handle only image tokens (text embedded in first 512 positions)

## 4. APPLICATIONS

This system enables:
- **Style Removal**: Remove artistic styles while preserving content
- **Safety Filtering**: Remove harmful content generation capabilities  
- **Copyright Protection**: Prevent generation of copyrighted characters/content
- **Bias Mitigation**: Reduce biased associations in generated content
- **Model Analysis**: Understand knowledge organization in diffusion transformers

The approach is model-agnostic and can be adapted to other transformer-based diffusion models
beyond Flux by modifying the attention processors and pipeline integration.
"""
