"""
Speculoos: Efficient speculative sampling for language models.

This package provides fast inference for large language models by using a smaller
draft model to propose tokens that are then verified by the target model.
"""

from .sampler import SpeculativeSampler

__version__ = "0.1.0"
__all__ = [
    "SpeculativeSampler",
    "speculative_sampling",
    "auto_regressive_sampling",
]


# Functional API wrappers for convenience
def speculative_sampling(
    input_text: str,
    draft_model_name: str,
    target_model_name: str,
    T: int,
    K: int = 5,
    eps: float = 1e-10,
    return_ids: bool = False
):
    """
    Generate text using speculative sampling (functional interface).
    
    This is a convenience function that creates a SpeculativeSampler and generates text.
    For multiple generations, consider creating a SpeculativeSampler instance directly
    to avoid reloading models.
    
    Args:
        input_text: Input text to continue from
        draft_model_name: Name or path of the draft model (e.g., "distilgpt2")
        target_model_name: Name or path of the target model (e.g., "gpt2")
        T: Number of new tokens to generate
        K: Number of speculative tokens per iteration (default: 5)
        eps: Epsilon for numerical stability (default: 1e-10)
        return_ids: If True, return token IDs instead of decoded text (default: False)
        
    Returns:
        Generated text string (or token IDs if return_ids=True)
        
    Example:
        >>> from speculoos import speculative_sampling
        >>> text = speculative_sampling(
        ...     "Once upon a time",
        ...     draft_model_name="distilgpt2",
        ...     target_model_name="gpt2",
        ...     T=20
        ... )
        >>> print(text)
    """
    sampler = SpeculativeSampler(draft_model_name, target_model_name, K=K, eps=eps)
    
    if return_ids:
        input_ids = sampler.encode(input_text)
        return sampler.sample(input_ids, T)
    else:
        return sampler.sample_and_decode(input_text, T)


def auto_regressive_sampling(
    input_text: str,
    model_name: str,
    T: int,
    return_ids: bool = False
):
    """
    Generate text using standard auto-regressive sampling (functional interface).
    
    This is a baseline comparison method that uses standard sequential token generation.
    For multiple generations, consider creating a SpeculativeSampler instance directly
    to avoid reloading models.
    
    Args:
        input_text: Input text to continue from
        model_name: Name or path of the model (e.g., "gpt2")
        T: Number of new tokens to generate
        return_ids: If True, return token IDs instead of decoded text (default: False)
        
    Returns:
        Generated text string (or token IDs if return_ids=True)
        
    Example:
        >>> from speculoos import auto_regressive_sampling
        >>> text = auto_regressive_sampling(
        ...     "Once upon a time",
        ...     model_name="gpt2",
        ...     T=20
        ... )
        >>> print(text)
    """
    # Use SpeculativeSampler with same model for both to get baseline
    sampler = SpeculativeSampler(model_name, model_name, K=1)
    
    if return_ids:
        input_ids = sampler.encode(input_text)
        return sampler.auto_regressive_sample(input_ids, T)
    else:
        return sampler.auto_regressive_sample_and_decode(input_text, T)
