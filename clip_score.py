# Import required libraries for CLIP-based evaluation
from transformers import CLIPProcessor, CLIPModel  # HuggingFace CLIP model and processor
import PIL                                         # Python Imaging Library for image handling


def get_clip_score(clip_model: CLIPModel, clip_processor: CLIPProcessor, text: str, image: PIL.Image):
    """
    Calculate CLIP similarity score between a text description and an image.
    
    CLIP (Contrastive Language-Image Pre-training) measures semantic similarity
    between text and images in a shared embedding space. Higher scores indicate
    better alignment between the text description and image content.
    
    This function is used to evaluate knowledge intervention effectiveness:
    - Lower scores after intervention = successful knowledge removal
    - Higher scores = knowledge still present in generated images
    
    Args:
        clip_model: Pre-trained CLIP model for computing embeddings
        clip_processor: CLIP processor for input preprocessing
        text: Text description to compare against the image
        image: PIL Image to evaluate
        
    Returns:
        float: Cosine similarity score between text and image embeddings (-1 to 1)
    """
    # Preprocess text and image inputs for the CLIP model
    # This handles tokenization, resizing, normalization, etc.
    inputs = clip_processor(text=text, images=image, return_tensors="pt", padding=True)

    # Move all input tensors to GPU for faster inference
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Compute CLIP embeddings for both text and image
    # This forward pass encodes inputs into a shared embedding space
    outputs = clip_model(**inputs)
    image_embeds = outputs.image_embeds  # Image embedding vector
    text_embeds = outputs.text_embeds    # Text embedding vector

    # Normalize embeddings to unit vectors for cosine similarity computation
    # L2 normalization ensures similarity is based on direction, not magnitude
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity via dot product of normalized embeddings
    # Result ranges from -1 (opposite) to 1 (identical semantic meaning)
    clip_score = (image_embeds @ text_embeds.T).item()

    return clip_score
