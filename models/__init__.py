from .vqgan_ae import VQGANAutoencoder
from .discriminator import PatchGANDiscriminator

MODEL_REGISTRY = {
    "VQGANAutoencoder": VQGANAutoencoder,
}

def get_model_class(model_name):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]