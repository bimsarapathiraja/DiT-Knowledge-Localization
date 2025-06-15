import argparse
import os
import glob
from csd import CSD_CLIP, convert_state_dict, get_csd_preprocess_transforms
import torch
from tqdm import tqdm
from PIL import Image


def load_csd_model():
    model = CSD_CLIP("vit_large", "default")
    model_path = "../pretrained_models/csd_checkpoint.pth"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = convert_state_dict(checkpoint['model_state_dict'])
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"=> loaded checkpoint with msg {msg}")

    return model


def calculate_directory_mean_csd_style_embedding(directory, model, csd_preprocess_transforms):
    output_file_path = os.path.join(directory, 'mean_csd_style_embedding.pt')
    if os.path.exists(output_file_path):
        print(f"[Warning] Mean CSD style embedding already exists at {output_file_path}. Skipping...")
        return

    image_paths = []
    for ext in ['png', 'jpg', 'jpeg']:
        image_paths.extend(glob.glob(os.path.join(directory, f'*.{ext}')))

    assert len(image_paths) > 0, f"No images found in {directory}"

    print(f"Found {len(image_paths)} images. Calculating mean CSD style embedding for {directory}...")

    style_features = torch.zeros(768).cuda()
    for image_path in tqdm(image_paths, desc="Calculating CSD for images"):
        image = Image.open(image_path).convert("RGB")
        x = csd_preprocess_transforms(image)
        with torch.no_grad():
            _, _, style_output = model(x.unsqueeze(0).cuda())
        style_features += style_output.squeeze()
    
    mean_style_embedding = style_features / len(image_paths)

    with open(output_file_path, 'wb') as f:
        torch.save(mean_style_embedding, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--eval_single_artist_directory",
        action="store_true",
    )
    parser.add_argument(
        "--artists_list_for_model",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    model = load_csd_model()
    csd_preprocess_transforms = get_csd_preprocess_transforms()

    if args.eval_single_artist_directory:
        calculate_directory_mean_csd_style_embedding(args.results_path, model, csd_preprocess_transforms)
    else:
        import sys
        sys.path.append("../")

        if args.artists_list_for_model is None:
            from dataset import get_artists_list
            artists = get_artists_list()
        else:
            from dataset import get_artists_list_for_model
            artists = get_artists_list_for_model(args.artists_list_for_model)

        for artist in tqdm(artists, leave=False, desc="Artists"):
            artist_path = os.path.join(args.results_path, artist)
            assert os.path.exists(artist_path), f"Artist directory {artist_path} does not exist"
            calculate_directory_mean_csd_style_embedding(artist_path, model, csd_preprocess_transforms)
