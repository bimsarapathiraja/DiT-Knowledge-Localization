import inspect
from typing import Callable, List, Optional, Dict, Union, Any

import numpy as np
import torch

from diffusers import FluxPipeline
from diffusers.pipelines.flux import FluxPipelineOutput
from diffusers.utils import is_torch_xla_available
from diffusers.image_processor import PipelineImageInput

import sys
sys.path.append("..")
from attention_processor import FluxEmbeddingModifierAttnProcessor


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

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
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class CustomFluxPipeline(FluxPipeline):
    def set_clean_scheduler(self):
        """
        Set the scheduler to the clean one
        """
        # Clone self.scheduler to self.clean_scheduler
        self.clean_scheduler = self.scheduler.__class__.from_config(self.scheduler.config)
    
    def set_embedding_modifier_attn_processor(self):
        for block in self.transformer.transformer_blocks:
            block.attn.set_processor(FluxEmbeddingModifierAttnProcessor())
    
        for block in self.transformer.single_transformer_blocks:
            block.attn.set_processor(FluxEmbeddingModifierAttnProcessor())

    def change_attn_embedding_modifier_mode(self, mm_attn_embedding_modifier_indices, single_attn_embedding_modifier_indices, mode):
        for idx in mm_attn_embedding_modifier_indices:
            self.transformer.transformer_blocks[idx].attn.processor.mode = mode

        for idx in single_attn_embedding_modifier_indices:
            self.transformer.single_transformer_blocks[idx].attn.processor.mode = mode
    
    def clear_attn_embedding_modifier_processors_cache(self):
        for block in self.transformer.transformer_blocks:
            block.attn.processor.clear_cache()
        
        for block in self.transformer.single_transformer_blocks:
            block.attn.processor.clear_cache()
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = 512, # Changed
        width: Optional[int] = 512, # Changed
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        # Added parameters
        clean_prompt: Union[str, List[str]] = None,
        mm_attn_embedding_modifier_indices: Optional[List[int]] = None,
        single_attn_embedding_modifier_indices: Optional[List[int]] = None,
        replace_pooled_prompt_embeds: bool = False,
        return_clean_image: bool = False,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        assert (mm_attn_embedding_modifier_indices is None) == (single_attn_embedding_modifier_indices is None), "If you want to use mm and single attn embedding modifier indices, please provide both or none"
        
        if mm_attn_embedding_modifier_indices is not None:
            assert len(mm_attn_embedding_modifier_indices) >= 0 and all([0 <= i < len(self.transformer.transformer_blocks) for i in mm_attn_embedding_modifier_indices])
            assert len(single_attn_embedding_modifier_indices) >= 0 and all([0 <= i < len(self.transformer.single_transformer_blocks) for i in single_attn_embedding_modifier_indices])
            assert clean_prompt is not None, "If you want to use clean pass, please provide a clean prompt"
        

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
        
        do_clean_pass = mm_attn_embedding_modifier_indices is not None
        if return_clean_image:
            assert do_clean_pass, "If you want to return clean image, please provide a clean prompt and mm_attn_embedding_modifier_indices"
        if do_clean_pass:
            self.set_embedding_modifier_attn_processor()
        
            (
                clean_prompt_embeds,
                clean_pooled_prompt_embeds,
                text_ids,
            ) = self.encode_prompt(
                prompt=clean_prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

            if replace_pooled_prompt_embeds:
                pooled_prompt_embeds = clean_pooled_prompt_embeds
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
