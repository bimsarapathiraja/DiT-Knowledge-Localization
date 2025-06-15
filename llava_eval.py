import argparse
import os
import glob
import json
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
from PIL import Image
from typing import List
from dataset import get_knowledge_dataset_class_and_get_list_fn


def extract_images_from_directory(directory_path: str) -> List[str]:
    image_paths = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths.extend(glob.glob(os.path.join(directory_path, f'*.{ext}')))
    
    return image_paths


class LLaVAEvaluator:
    def __init__(self, processor: AutoProcessor, model: LlavaForConditionalGeneration):
        self.processor = processor
        self.model = model
    
    def eval_image_text_pair(self, image: Image, knowledge: str, knowledge_type: str):
        assert knowledge_type in ["style", "place", "copyright", "animal", "celebrity", "safety"], f"Invalid knowledge type: {knowledge_type}"

        if knowledge_type == "style":
            print("[Info] Using artistic style knowledge type for questioning")
            question = f"Is the image in the artistic style of {knowledge} (artist)? just say yes or no"
        else:
            print(f"[Info] Using {knowledge_type} knowledge type for questioning")
            question = f"Is {knowledge} in the image? just say yes or no"

        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image"},
                ],
        },]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

        output = self.processor.decode(output[0], skip_special_tokens=True).split("ASSISTANT: ")[-1].strip().lower()
        if output == "yes":
            return True
        elif output == "no":
            return False
        else:
            return None


    def eval_knowledge_directory(self, directory_path: str, knowledge: str, knowledge_type: str):
        """Evaluate a directory of images with a specific knowledge."""

        output_file_path = os.path.join(directory_path, 'llava_results.json')
        if os.path.exists(output_file_path):
            print(f"[Warning] LLaVA evaluation results already exists at {output_file_path}. Skipping...")
            return

        image_paths = extract_images_from_directory(directory_path)

        if len(image_paths) == 0:
            print(f"[Warning] No images found in {directory_path}. Skipping")
            return

        print(f"Found {len(image_paths)} images. Running LLaVA evaluation for {directory_path} [evaluation knowledge={knowledge}]...")
        res = {}
        for image_path in tqdm(image_paths, desc="LLaVA evaluation for images"):
            res[os.path.basename(image_path)] = self.eval_image_text_pair(Image.open(image_path), knowledge, knowledge_type)
        
        with open(output_file_path, 'w') as f:
            json.dump({"knowledge": knowledge, "results": res}, f, indent=4)
    

    def eval_no_knowledge_directory(self, directory_path: str, knowledge_list: List[str], knowledge_type: str):
        """Evaluate a directory of images with no knowledge (evaluation across all knowledge list)."""

        output_file_path = os.path.join(directory_path, 'llava_results.json')
        if os.path.exists(output_file_path):
            print(f"[Warning] LLaVA evaluation results already exists at {output_file_path}. Skipping...")
            return

        image_paths = extract_images_from_directory(directory_path)

        if len(image_paths) == 0:
            print(f"[Warning] No images found in {directory_path}. Skipping")
            return

        print(f"Found {len(image_paths)} images. Running LLaVA evaluation for {directory_path} [evaluation knowledge list length = # {len(knowledge_list)}]...")
        res = {}
        for knowledge in knowledge_list:
            res[knowledge] = {}
            for image_path in tqdm(image_paths, desc=f"LLaVA evaluation for images with knowledge: {knowledge}"):
                res[knowledge][os.path.basename(image_path)] = self.eval_image_text_pair(Image.open(image_path), knowledge, knowledge_type)
        
        with open(output_file_path, 'w') as f:
            json.dump(res, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        required=True,
        choices=["eval_all_knowledge_directories", "eval_a_single_no_knowledge_directory"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["pixart", "flux", "sana"],
    )
    parser.add_argument(
        "--knowledge_type",
        type=str,
        required=True,
        choices=["style", "place", "copyright", "animal", "celebrity", "safety"]
    )
    args = parser.parse_args()

    assert os.path.exists(args.results_path), f"Results path {args.results_path} does not exist"

    return args


if __name__ == "__main__":
    args = parse_args()

    llava_evaluator = LLaVAEvaluator(
        AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", use_fast=True),
        LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    )

    _, get_knowledge_list_fn = get_knowledge_dataset_class_and_get_list_fn(args.knowledge_type, args.model_name)

    if args.eval_type == "eval_all_knowledge_directories":
        knowledge_list = get_knowledge_list_fn()
        for knowledge in knowledge_list:
            llava_evaluator.eval_knowledge_directory(os.path.join(args.results_path, knowledge), knowledge, args.knowledge_type)
    elif args.eval_type == "eval_a_single_no_knowledge_directory":
        llava_evaluator.eval_no_knowledge_directory(args.results_path, get_knowledge_list_fn(), args.knowledge_type)

    print("Done!")
