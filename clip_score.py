from transformers import CLIPProcessor, CLIPModel
import PIL


def get_clip_score(clip_model: CLIPModel, clip_processor: CLIPProcessor, text: str, image: PIL.Image):
    inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True)

    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Get CLIP embeddings
    outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # Normalize embeddings
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity
    clip_score = (image_embeds @ text_embeds.T).item()

    return clip_score
