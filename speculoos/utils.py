
import torch
from transformers import AutoModelForCausalLM

def sample_random(probabilities: torch.Tensor, temperature: float = 1.0,
                  top_k: int = None, top_p: float = None) -> torch.Tensor:
    """
    Sample from probability distribution with optional temperature, top-k, top-p.
    
    Args:
        probabilities: Probability distribution tensor of shape (batch_size, vocab_size)
        temperature: Temperature for sampling (default: 1.0)
        top_k: Top-k filtering value (default: None)
        top_p: Top-p (nucleus) filtering value (default: None)
        
    Returns:
        Sampled token index tensor of shape (batch_size, 1)
    """

    # Apply temperature
    if temperature != 1.0:
        probabilities = torch.pow(probabilities, 1.0 / temperature)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    
    # Apply top-k filtering
    if top_k is not None:
        top_k_probs, top_k_indices = torch.topk(probabilities, min(top_k, probabilities.size(-1)))
        probabilities = torch.zeros_like(probabilities).scatter_(-1, top_k_indices, top_k_probs)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    
    # Apply top-p (nucleus) filtering
    if top_p is not None:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum_probs <= top_p
        mask[..., 0] = True  # Always keep at least one token
        sorted_probs[~mask] = 0.0
        probabilities = torch.zeros_like(probabilities).scatter_(-1, sorted_indices, sorted_probs)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    
    # Sample
    return torch.multinomial(probabilities, num_samples=1)

def relu_n(x: torch.Tensor) -> torch.Tensor:
    """
    Computes the ReLU function and applies normalization.
    
    Args:
        x: Input tensor
        
    Returns:
        Normalized ReLU output
    """
    x_relu = torch.relu(x)
    return x_relu / torch.sum(x_relu)

def predict(model: AutoModelForCausalLM, x: torch.Tensor) -> torch.Tensor:
    """
    Gets the probability distribution over the next token given the input sequence x using the model.
    
    Args:
        model: The language model to use for prediction
        x: The input sequence (token IDs)
        
    Returns:
        Probability distribution over the next token of shape (batch_size, vocab_size)
    """
    # Get logits from the model
    with torch.no_grad():
        outputs = model(x)
        logits = outputs.logits
    
    # Sample next token from the last position
    next_token_logits = logits[:, -1, :]
    return torch.softmax(next_token_logits, dim=-1)
