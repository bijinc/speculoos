from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import time

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


def auto_regressive_sampling(model: AutoModelForCausalLM, x: torch.Tensor, T: int) -> torch.Tensor:
    """
    Auto-regressive sampling from the target model for T steps.
        - model: the target language model to sample from
        - x: the initial input sequence (token IDs)
        - T: the number of tokens to sample
    Returns the generated sequence of token IDs after T steps.
    """
    for _ in range(T):
        # get probabilities
        probabilities = predict(model, x)

        # sample next token from the target model
        predicted_token = sample_random(probabilities)
        
        # Concatenate the new token
        x = torch.cat([x, predicted_token], dim=-1)

    return x

def speculative_sampling(
        target_model: AutoModelForCausalLM,
        draft_model: AutoModelForCausalLM,
        x: torch.Tensor,
        K: int,
        T: int,
        eps=1e-10
    ) -> torch.Tensor:
    """
    Speculative sampling from the target model using a draft model for K steps.
        - target_model: the target language model to sample from
        - draft_model: the draft language model used for proposing tokens
        - x: the initial input sequence (token IDs)
        - K: the number of speculative steps to perform
        - T: the total number of tokens to sample (including speculative steps)
    Returns the generated sequence of token IDs after T steps.
    """
    n = x.shape[-1]
    T += n

    while n < T:
        # Save the starting position for this iteration
        n_start = n
        
        # Drafting: auto-regressive sampling using the draft model
        x_draft = x.clone()
        draft_probs = []
        
        for _ in range(K):
            # get probabilities
            p = predict(draft_model, x_draft)
            draft_probs.append(p)
            
            # sample next token from the draft model
            predicted_token = sample_random(p)
            
            # Concatenate the new token
            x_draft = torch.cat([x_draft, predicted_token], dim=-1)
        
        # Verification: get target model probabilities for entire draft sequence
        with torch.no_grad():
            outputs = target_model(x_draft)
            logits = outputs.logits
        
        target_probs = []
        for k in range(K + 1):
            pos = n_start - 1 + k
            next_token_logits = logits[:, pos, :]
            target_probs.append(torch.softmax(next_token_logits, dim=-1))

        # Correction: accept or reject predicted tokens
        all_accepted = True
        for k in range(K):
            j = x_draft[:, n_start + k]  # Token at position n_start+k
            
            p_j = draft_probs[k][0, j.item()]  # Draft probability for token j
            q_j = target_probs[k][0, j.item()]  # Target probability for token j

            r = random.random()
            if r < min(1.0, (q_j / (p_j + eps)).item()):
                # token accepted
                x = torch.cat([x, j.unsqueeze(0)], dim=-1)
                n += 1
            else:
                # token rejected, resample
                adjusted_probs = relu_n(target_probs[k][0] - draft_probs[k][0])
                resampled_token = sample_random(adjusted_probs.unsqueeze(0))
                x = torch.cat([x, resampled_token], dim=-1)
                n += 1
                all_accepted = False
                break
        
        if all_accepted:
            # sample an extra token from target model at the last position
            x = torch.cat([x, sample_random(target_probs[-1])], dim=-1)
            n += 1
    
    return x

def decode(output_ids: torch.Tensor, tokenizer: AutoTokenizer) -> str:
    """
    Decodes the generated token IDs into text using the model's tokenizer.
        - output_ids: the sequence of generated token IDs
        - model: the language model whose tokenizer will be used for decoding
    Returns the decoded text string.
    """
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def run_auto_regressive_sampling(input_seq: str, model_name: str = "gpt2", T: int = 20):
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    input_ids = tokenizer.encode(input_seq, return_tensors="pt")

    print(f"\nRunning auto-regressive sampling using {model_name}")
    start = time.perf_counter()

    output_ids = auto_regressive_sampling(model, input_ids, T)
    text = decode(output_ids, tokenizer)

    elapsed_time = time.perf_counter() - start

    print(f"Output: {text}")
    print(f"Time: {elapsed_time:.2f}s")


def run_speculative_sampling(
        input_seq: str,
        draft_model_name: str = "distilgpt2",
        target_model_name: str = "gpt2", 
        K: int = 5, 
        T: int = 20
    ):
    
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name)
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    
    input_ids = tokenizer.encode(input_seq, return_tensors="pt")

    print(f"\nRunning speculative sampling using {target_model_name} with draft model {draft_model_name}")
    start = time.perf_counter()

    output_ids = speculative_sampling(target_model, draft_model, input_ids, K, T)
    text = decode(output_ids, tokenizer)

    elapsed_time = time.perf_counter() - start

    print(f"Output: {text}")
    print(f"Time: {elapsed_time:.2f}s")


def main():

    input_seq = "Once upon a time"

    run_auto_regressive_sampling(input_seq, model_name="gpt2", T=20)

    run_speculative_sampling(input_seq, draft_model_name="distilgpt2", target_model_name="gpt2", K=5, T=20) 


if __name__ == "__main__":
    main()