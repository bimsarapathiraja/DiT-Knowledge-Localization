import torch
from transformers import PreTrainedTokenizer


def find_substring_token_indices(prompt: str, substr: str, tokenizer: PreTrainedTokenizer, model="pixart"):
    assert model in ["pixart", "sana", "flux"], f"Model {model} not supported"

    prompt_tokens = tokenizer(prompt).input_ids

    if model == "pixart":
        substr_tokens = tokenizer(substr).input_ids[:-1]
    elif model == "sana":
        if "in the style of" in prompt: # TODO: Clean code [for sana, " X" and "X" resul in different tokens]
            substr = " " + substr
        else:
            substr = substr
        substr_tokens = tokenizer(substr).input_ids[1:]
    elif model == "flux":
        substr_tokens = tokenizer(substr).input_ids[1:-1]
    else:
        raise ValueError(f"Model {model} not recognized.")

    start_idx = -1
    for i in range(len(prompt_tokens) - len(substr_tokens) + 1):
        if prompt_tokens[i:i+len(substr_tokens)] == substr_tokens:
            start_idx = i
            break   

    assert start_idx != -1, "substr_tokens not found in tokens"

    token_indices = list(range(start_idx, start_idx + len(substr_tokens)))

    if tokenizer.decode([prompt_tokens[token_idx] for token_idx in token_indices]) != substr:
        print("============================ Warning ============================")
        print("[Warning] tokenizer.decode([prompt_tokens[token_idx] for token_idx in token_indices]) != substr")
        print(f"[Warning] Decoded: {tokenizer.decode([prompt_tokens[token_idx] for token_idx in token_indices])}")
        print(f"[Warning] Expected: {substr}")
        print("=================================================================")

    return token_indices


def latents_to_images(pipe, latents):
    latents = latents.to(pipe.vae.dtype)
    with torch.no_grad():
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    images = pipe.image_processor.postprocess(images, output_type="pil")
    
    return images


def get_worker_list_chunk(arr, num_workers, worker_idx, print_log=True):
    arr_len = len(arr)

    chunk_size = (arr_len + num_workers - 1) // num_workers

    start_index = chunk_size * worker_idx
    end_index = min((worker_idx + 1) * chunk_size, arr_len)

    if print_log:
        print(f"Choosing chunk ({start_index}:{end_index})")
        print(f"First prompt of the chunk: \"{arr[start_index]}\"")
        print(f"Last prompt of the chunk: \"{arr[end_index-1]}\"")

    return arr[start_index:end_index]


def print_arguments(args):
    print("===================== Arguments =====================")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=====================================================")
