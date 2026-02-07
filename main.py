from transformers import AutoModelForCausalLM
import torch
import random

def sample_random(p):
    return torch.multinomial(p, num_samples=1)
    # return np.random.choice(np.arange(p.shape[-1]), p=p)

def relu_n(x):
    """Computes the ReLU function and applies normalization"""
    x_relu = torch.max(x, 0)
    return x_relu / torch.sum(x_relu)

def predict(model, x):
    # Get logits from the model
    with torch.no_grad():
        outputs = model(x)
        logits = outputs.logits
    
    # Sample next token from the last position
    next_token_logits = logits[:, -1, :]
    return torch.softmax(next_token_logits, dim=-1)


def auto_regressive_sampling(target_model: AutoModelForCausalLM, x, T):
    # t = 1
    for _ in range(T):
        # get probabilities
        probabilities = predict(target_model, x)

        # sample next token from the target model
        predicted_token = sample_random(probabilities)
        
        # Concatenate the new token
        x = torch.cat([x, predicted_token], dim=-1)
        # t += 1

    return x

def speculative_sampling(target_model, draft_model, x, K, T):
    n = x.shape[-1]
    T += n

    while n < T:
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
        target_probs = []
        for k in range(K + 1):
            q = predict(target_model, x[:, :n+k])  # Get probs up to position n+k
            target_probs.append(q)

        # Correction: accept or reject predicted tokens
        all_accepted = True
        for k in range(K):
            j = x_draft[:, n + k]  # Token at position n+k
            
            p_j = draft_probs[k][0, j.item()]  # Draft probability for token j
            q_j = target_probs[k][0, j.item()]  # Target probability for token j

            r = random.random()
            if r < min(1.0, (q_j / p_j).item()):
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
