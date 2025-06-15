import argparse
from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm

import os
import sys
from pathlib import Path
import inspect

sys.path.append(str(Path(__file__).resolve().parent.parent))

from clip_score import get_clip_score
from dataset import get_knowledge_dataset_class_and_get_list_fn, get_eval_text_for_knowledge
from pipelines.loader import load_pipe

from functools import partial

import json


def generate_and_evaluate(args, pipe, clip_score_fn, dataset, knowledge_list):
    res = {knowledge: {} for knowledge in knowledge_list}

    for prompt in tqdm(dataset, desc=f"Generating images"):
        prompt = dataset.get_clean_prompt(prompt)

        for i in range(args.num_images_per_eval_prompt):
            seed = args.generator_seed + i

            image = pipe(
                prompt=prompt,
                generator=torch.Generator().manual_seed(seed),
            ).images[0].resize((512, 512))

            file_name = f"{prompt}_{seed}.png"

            for knowledge in knowledge_list:
                res[knowledge][file_name] = clip_score_fn(get_eval_text_for_knowledge(args.knowledge_type, knowledge), image)

            image.save(f"{args.results_path}/{file_name}")


    with open(f"{args.results_path}/results.json", "w") as f:
        json.dump(res, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generator_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--results_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--num_images_per_eval_prompt",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default=None,
        choices=["pixart", "sana", "flux"],
    )
    parser.add_argument(
        "--knowledge_type",
        type=str,
        required=True,
        default=None,
        choices=["style", "place", "copyright", "animal", "celebrity", "safety"]
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    pipe = load_pipe(model_name=args.model)

    if not os.path.exists(args.results_path):
        print(f"Creating path for {args.results_path}")
        os.makedirs(args.results_path, exist_ok=True)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

    dataset_cls, get_knowledge_list_fn = get_knowledge_dataset_class_and_get_list_fn(args.knowledge_type)

    generate_and_evaluate(args, pipe, partial(get_clip_score, clip_model, clip_processor), dataset_cls('XXXXX', "both"), get_knowledge_list_fn())
