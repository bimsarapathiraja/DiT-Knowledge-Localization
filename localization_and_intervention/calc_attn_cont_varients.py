import torch
from typing import Callable
from functools import partial
from diffusers import FluxPipeline
import sys
sys.path.append("..")
from attention_processor import FluxCachingAttnProcessor


# TODO: Ensure that only a single image is being generated (since we are using [1] for getting the guided latent)
def _calc_attn_cont(args, pipe, token_indices, calc_attn_cont_per_block_and_timestep_func: Callable):
    attn_cont = torch.zeros(len(pipe.transformer.transformer_blocks), args.num_timesteps)

    for t in range(args.num_timesteps):
        for b in range(len(pipe.transformer.transformer_blocks)):
            if isinstance(pipe, FluxPipeline):
                attn = pipe.transformer.transformer_blocks[b].attn
            else:
                attn = pipe.transformer.transformer_blocks[b].attn2
            assert len(attn.processor.attention_maps) == len(attn.processor.values) == args.num_timesteps
            attn_cont[b][t] += calc_attn_cont_per_block_and_timestep_func(attn, token_indices, t)
    
    return attn_cont


@torch.no_grad()
def calc_attn_cont_using_m_per_block_and_t_fn(attn, token_indices, t):
    m = attn.processor.attention_maps[t][:, :, token_indices].to("cuda")
    return torch.mean(torch.norm(attn.batch_to_head_dim(m)[1], dim=1)).cpu()


@torch.no_grad()
def calc_attn_cont_using_m_mult_v_per_block_and_t_fn(attn, token_indices, t):
    if isinstance(attn.processor, FluxCachingAttnProcessor):
        m = attn.processor.attention_maps[t][0, :, :, token_indices].to("cuda") # (24, 1024, # token indices) [for 512x512 image]
        v = attn.processor.values[t][0, :, token_indices, :].to("cuda") # (24, # token indices, 128)
        o = (m @ v).transpose(0, 1).reshape(m.shape[1], v.shape[0] * v.shape[2]) # (24, 1024, 128) -> (1024, 24*128)
        return torch.norm(o, dim=1).mean().cpu()
    
    m = attn.processor.attention_maps[t][:, :, token_indices].to("cuda")
    v = attn.processor.values[t][:, token_indices, :].to("cuda")
    return torch.mean(torch.norm(attn.batch_to_head_dim(m @ v)[1], dim=1)).cpu()


@torch.no_grad()
def calc_attn_cont_using_m_mult_v_mult_o_per_block_and_t_fn(attn, token_indices, t):
    if isinstance(attn.processor, FluxCachingAttnProcessor):
        m = attn.processor.attention_maps[t][0, :, :, token_indices].to("cuda") # (24, 1024, # token indices) [for 512x512 image]
        v = attn.processor.values[t][0, :, token_indices, :].to("cuda") # (24, # token indices, 128)
        o = (m @ v).transpose(0, 1).reshape(m.shape[1], v.shape[0] * v.shape[2]) # (24, 1024, 128) -> (1024, 24*128)
        o = attn.to_out[0](o) # (1024, 24*128) -> (1024, 24*128)
        return torch.norm(o, dim=1).mean().cpu()
    
    m = attn.processor.attention_maps[t][:, :, token_indices].to("cuda")
    v = attn.processor.values[t][:, token_indices, :].to("cuda")
    return torch.mean(torch.norm(attn.to_out[0](attn.batch_to_head_dim(m @ v)[1]), dim=1)).cpu()


@torch.no_grad()
def calc_attn_cont_using_m_mult_v_mult_o_normalized_by_out_norm_per_block_and_t_fn(attn, token_indices, t):
    m = attn.processor.attention_maps[t][:, :, token_indices].to("cuda")
    v = attn.processor.values[t][:, token_indices, :].to("cuda")
    return torch.mean(torch.norm(attn.to_out[0](attn.batch_to_head_dim(m @ v)[1]), dim=1) / attn.processor.out_norms[t].to("cuda")).cpu()


@torch.no_grad()
def calc_attn_cont_using_m_mult_v_mult_o_relative_impact_per_block_and_t_fn(attn, token_indices, t):
    m = attn.processor.attention_maps[t][:, :, token_indices].to("cuda")
    v = attn.processor.values[t][:, token_indices, :].to("cuda")

    o = attn.to_out[0](attn.batch_to_head_dim(m @ v)[1])
    i = attn.processor.in_norms[t].to("cuda")
    relative_impact = torch.abs(torch.norm(o - i, dim=1) / torch.norm(i, dim=1))

    return torch.mean(relative_impact).cpu()


calc_attn_cont_function_variants_mapping = {
    "m": calc_attn_cont_using_m_per_block_and_t_fn,
    "m_mult_v": calc_attn_cont_using_m_mult_v_per_block_and_t_fn,
    "m_mult_v_mult_o": calc_attn_cont_using_m_mult_v_mult_o_per_block_and_t_fn,
    "m_mult_v_mult_o_normalized_by_out_norm": calc_attn_cont_using_m_mult_v_mult_o_normalized_by_out_norm_per_block_and_t_fn,
    "m_mult_v_mult_o_relative_impact": calc_attn_cont_using_m_mult_v_mult_o_relative_impact_per_block_and_t_fn,
}


def get_calc_attn_cont_fn(variant: str):
    assert variant in calc_attn_cont_function_variants_mapping, f"Invalid variant: {variant}"

    return partial(
        _calc_attn_cont,
        calc_attn_cont_per_block_and_timestep_func=calc_attn_cont_function_variants_mapping[variant]      
    )
