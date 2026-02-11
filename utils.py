import torch
from transformers import AutoModelForCausalLM

def sample_random(p: torch.Tensor) -> torch.Tensor:
    """Samples a token index from the given probability distribution p."""
    return torch.multinomial(p, num_samples=1)

def relu_n(x: torch.Tensor) -> torch.Tensor:
    """Computes the ReLU function and applies normalization"""
    x_relu = torch.relu(x)
    return x_relu / torch.sum(x_relu)

def predict(model: AutoModelForCausalLM, x: torch.Tensor) -> torch.Tensor:
    """
    Gets the probability distribution over the next token given the input sequence x using the model.
        - model: the language model to use for prediction
        - x: the input sequence (token IDs)
    Returns the probability distribution over the next token.
    """
    # Get logits from the model
    with torch.no_grad():
        outputs = model(x)
        logits = outputs.logits
    
    # Sample next token from the last position
    next_token_logits = logits[:, -1, :]
    return torch.softmax(next_token_logits, dim=-1)