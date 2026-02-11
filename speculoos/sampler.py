
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import predict, sample_random, relu_n


class SpeculativeSampler:
    """
    A sampler that uses speculative sampling to accelerate language model inference.
    
    Speculative sampling uses a smaller draft model to propose multiple tokens at once,
    which are then verified by the target model. This can provide significant speedups
    while maintaining the same output distribution as auto-regressive sampling.
    
    Args:
        draft_model_name: Name or path of the draft model (e.g., "distilgpt2")
        target_model_name: Name or path of the target model (e.g., "gpt2")
        K: Number of speculative tokens to generate per iteration (default: 5)
        eps: Small epsilon value for numerical stability (default: 1e-10)
        
    Example:
        >>> sampler = SpeculativeSampler("distilgpt2", "gpt2", K=5)
        >>> text = sampler.sample_and_decode("Once upon a time", T=20)
        >>> print(text)
    """
    
    def __init__(self, draft_model_name: str, target_model_name: str, K: int = 5, eps: float = 1e-10, device: str = None):
        """Initialize the speculative sampler with draft and target models."""
        self.draft_model_name = draft_model_name
        self.target_model_name = target_model_name
        self.K = K
        self.eps = eps
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load models and tokenizer
        self.draft_model = self._load_model(draft_model_name)
        self.target_model = self._load_model(target_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    
    def _load_model(self, model_name: str) -> AutoModelForCausalLM:
        """Load a causal language model from Hugging Face."""
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
            model = model.to(self.device)
            return model
        except Exception as e:
            raise ValueError(f"Error loading model {model_name}: {e}")
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode input text into token IDs using the tokenizer.
        
        Args:
            text: Input text string
            
        Returns:
            Token IDs tensor of shape (1, seq_len)
        """
        return self.tokenizer.encode(text, return_tensors="pt").to(self.device)
    
    def decode(self, output_ids: torch.Tensor) -> str:
        """
        Decode token IDs into text using the tokenizer.
        
        Args:
            output_ids: Token IDs tensor of shape (1, seq_len)
            
        Returns:
            Decoded text string
        """
        return self.tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True)
    
    def sample(self, input_ids: torch.Tensor, T: int) -> torch.Tensor:
        """
        Generate T tokens using speculative sampling.
        
        Args:
            input_ids: Initial input sequence token IDs of shape (1, seq_len)
            T: Number of new tokens to generate
            
        Returns:
            Generated sequence token IDs of shape (1, seq_len + T)
        """
        n = input_ids.shape[-1]
        T_total = T + n
        
        # Track acceptance stats
        total_drafted = 0
        total_accepted = 0

        while n < T_total:
            n_start = n
            
            # Drafting: auto-regressive sampling using the draft model
            x_draft = input_ids.clone()
            draft_probs = []
            
            for _ in range(self.K):
                # Get probabilities from draft model
                p = predict(self.draft_model, x_draft)
                draft_probs.append(p)
                
                # Sample next token from the draft model
                predicted_token = sample_random(p)
                
                # Concatenate the new token
                x_draft = torch.cat([x_draft, predicted_token], dim=-1)
            
            # Verification: get target model probabilities for entire draft sequence
            with torch.no_grad():
                outputs = self.target_model(x_draft)
                logits = outputs.logits
            
            target_probs = []
            for k in range(self.K + 1):
                pos = n_start - 1 + k
                next_token_logits = logits[:, pos, :]
                target_probs.append(torch.softmax(next_token_logits, dim=-1))

            # Correction: accept or reject predicted tokens
            all_accepted = True
            num_accepted = 0
            for k in range(self.K):
                j = x_draft[:, n_start + k]  # Token at position n_start+k
                
                p_j = draft_probs[k][0, j.item()]  # Draft probability for token j
                q_j = target_probs[k][0, j.item()]  # Target probability for token j

                r = random.random()
                if r < min(1.0, (q_j / (p_j + self.eps)).item()):
                    # Token accepted
                    input_ids = torch.cat([input_ids, j.unsqueeze(0)], dim=-1)
                    n += 1
                    num_accepted += 1
                else:
                    # Token rejected, resample from adjusted distribution
                    adjusted_probs = relu_n(target_probs[k][0] - draft_probs[k][0])
                    resampled_token = sample_random(adjusted_probs.unsqueeze(0))
                    input_ids = torch.cat([input_ids, resampled_token], dim=-1)
                    n += 1
                    all_accepted = False
                    break
            
            if all_accepted:
                # Sample an extra token from target model at the last position
                input_ids = torch.cat([input_ids, sample_random(target_probs[-1])], dim=-1)
                n += 1
                num_accepted += 1
            
            total_drafted += self.K
            total_accepted += num_accepted
        
        # if total_drafted > 0:
        #     acceptance_rate = total_accepted / total_drafted
        #     print(f"  Acceptance rate: {acceptance_rate:.1%} ({total_accepted}/{total_drafted} tokens)")
        
        return input_ids
    
    def auto_regressive_sample(self, input_ids: torch.Tensor, T: int) -> torch.Tensor:
        """
        Generate T tokens using standard auto-regressive sampling (baseline).
        
        Args:
            input_ids: Initial input sequence token IDs of shape (1, seq_len)
            T: Number of new tokens to generate
            
        Returns:
            Generated sequence token IDs of shape (1, seq_len + T)
        """
        x = input_ids.clone()
        
        for _ in range(T):
            # Get probabilities from target model
            probabilities = predict(self.target_model, x)
            
            # Sample next token
            predicted_token = sample_random(probabilities)
            
            # Concatenate the new token
            x = torch.cat([x, predicted_token], dim=-1)
        
        return x
    
    def sample_and_decode(self, input_text: str, T: int) -> str:
        """
        Encode input text, generate tokens, and decode in one step.
        
        Args:
            input_text: Input text string
            T: Number of new tokens to generate
            
        Returns:
            Generated text string
        """
        input_ids = self.encode(input_text)
        output_ids = self.sample(input_ids, T)
        return self.decode(output_ids)
    
    def auto_regressive_sample_and_decode(self, input_text: str, T: int) -> str:
        """
        Encode input text, generate tokens using auto-regressive sampling, and decode.
        
        Args:
            input_text: Input text string
            T: Number of new tokens to generate
            
        Returns:
            Generated text string
        """
        input_ids = self.encode(input_text)
        output_ids = self.auto_regressive_sample(input_ids, T)
        return self.decode(output_ids)
