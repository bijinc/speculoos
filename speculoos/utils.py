
import torch
from transformers import AutoModelForCausalLM

def sample_random(p: torch.Tensor) -> torch.Tensor:
    """
    Samples a token index from the given probability distribution p.
    
    Args:
        p: Probability distribution tensor of shape (batch_size, vocab_size)
        
    Returns:
        Sampled token index tensor of shape (batch_size, 1)
    """
    return torch.multinomial(p, num_samples=1)

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
