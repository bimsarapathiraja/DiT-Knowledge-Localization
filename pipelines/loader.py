import torch


def load_pipe(model_name):
    if model_name == "pixart":
        from pipelines.pixart_alpha import CustomPixArtAlphaPipeline

        pipe = CustomPixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to("cuda")
        pipe.set_custom_attn_processor()
    elif model_name == "sana":
        from pipelines.sana import CustomSanaPipeline
        
        pipe = CustomSanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", torch_dtype=torch.float32
        ).to("cuda")
        pipe.text_encoder.to(torch.bfloat16)
        pipe.transformer = pipe.transformer.to(torch.bfloat16)
    elif model_name == "flux":
        from pipelines.flux import CustomFluxPipeline
        
        pipe = CustomFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        ).to("cuda")
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return pipe
