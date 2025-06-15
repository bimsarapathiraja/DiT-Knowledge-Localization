import argparse
import datetime
from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm

import os
import sys
from pathlib import Path
import inspect

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import get_worker_list_chunk
from clip_score import get_clip_score
from dataset import get_knowledge_dataset_class_and_get_list_fn, get_eval_text_for_knowledge
from pipelines.loader import load_pipe

from functools import partial

import json


def generate_and_evaluate(args, pipe, clip_score_fn, dataset):
    clip_score_results_path = f"{args.results_path}/{dataset.knowledge}/results.json"
    if os.path.exists(clip_score_results_path):
        print(f"[INFO - Skipping Generation] Results for \"{dataset.knowledge}\" already exist. Skipping...")
        return

    res = {}
    for prompt in tqdm(dataset, desc=f"Evaluating {args.knowledge_type.title()} {dataset.knowledge}"):
        for i in range(args.num_images_per_eval_prompt):
            seed = args.generator_seed + i

            image = pipe(
                prompt=prompt,
                generator=torch.Generator().manual_seed(seed),
            ).images[0].resize((512, 512))

            file_name = f"{prompt}_{seed}.png"

            if not os.path.exists(f"{args.results_path}/{dataset.knowledge}"):
                os.makedirs(f"{args.results_path}/{dataset.knowledge}")
            image.save(f"{args.results_path}/{dataset.knowledge}/{file_name}")
            
            res[file_name] = clip_score_fn(get_eval_text_for_knowledge(args.knowledge_type, dataset.knowledge), image)
    
    with open(clip_score_results_path, "w") as f:
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
        "--num_workers",
        type=int,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--worker_idx",
        type=int,
        default=None,
        required=True,
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
        choices=["style", "place", "animal", "celebrity", "copyright", "safety"]
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    dataset_class, get_knowledge_list_fn = get_knowledge_dataset_class_and_get_list_fn(args.knowledge_type)

    worker_knowledge_list = get_worker_list_chunk(get_knowledge_list_fn(), args.num_workers, args.worker_idx)

    pipe = load_pipe(model_name=args.model)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to("cuda")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)

    for knowledge in worker_knowledge_list:
        generate_and_evaluate(args, pipe, partial(get_clip_score, clip_model, clip_processor), dataset_class(knowledge, "both"))
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[Completed {current_time}] {args.knowledge_type.title()}: {knowledge}')
