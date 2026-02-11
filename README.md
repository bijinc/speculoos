# Speculoos 🍪

**Efficient speculative sampling for language models**

Speculoos is a Python package that accelerates language model inference using speculative sampling. It uses a smaller draft model to propose multiple tokens at once, which are then verified by the target model. This provides significant speedups while maintaining the exact same output distribution as standard auto-regressive sampling.

## What is Speculative Sampling?

Speculative sampling is an inference optimization technique that:
- Uses a fast "draft" model to propose K tokens ahead
- Validates these proposals with the target model in parallel
- Accepts or rejects tokens based on probability matching
- Achieves 2-3x speedup with no quality loss

The key insight is that while a single forward pass through a large model is expensive, verifying multiple tokens in parallel is much faster than generating them one at a time.

⚠️ **Hardware Requirements**: Speculative sampling is optimized for **GPU inference**. On CPU, the overhead of multiple model calls typically outweighs any benefits. For best results, use a CUDA-enabled GPU.

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/bijinc/speculoos.git
cd speculoos

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### From PyPI (Coming Soon)

```bash
pip install speculoos
```

## Quick Start
Example code

```python
from speculoos import SpeculativeSampler

# Create a sampler with draft and target models
sampler = SpeculativeSampler(
    draft_model_name="gpt2",
    target_model_name="gpt2-large",
    K=5  # Number of speculative tokens per iteration
)

# Generate text
text = sampler.sample_and_decode("Once upon a time", T=20)
print(text)

# Compare with baseline auto-regressive sampling
baseline_text = sampler.auto_regressive_sample_and_decode("Once upon a time", T=20)
print(baseline_text)
```

## Advanced Usage

### Working with Token IDs

```python
from speculoos import SpeculativeSampler

sampler = SpeculativeSampler("distilgpt2", "gpt2-medium")

# Encode input text to token IDs
input_ids = sampler.encode("Once upon a time")

# Generate tokens
output_ids = sampler.sample(input_ids, T=20)

# Decode back to text
text = sampler.decode(output_ids)
print(text)
```

### Tuning Performance

The `K` parameter controls the number of speculative tokens generated per iteration:
- **Higher K** (e.g., 8-10): More tokens proposed, but lower acceptance rate
- **Lower K** (e.g., 3-5): Fewer tokens proposed, but higher acceptance rate
- **Optimal K**: Typically 4-6 for most model pairs

```python
# Experiment with different K values
for k in [3, 5, 7, 10]:
    sampler = SpeculativeSampler("distilgpt2", "gpt2-medium", K=k)
    # ... benchmark performance
```

### Choosing Model Pairs

For best results:
- **Draft model**: Should be 2-4x smaller/faster than target model
- **Same model family**: Use consecutive sizes from same architecture (not distilled versions)
- **Compatible tokenizers**: Use models with the same tokenizer

Good pairs (high acceptance rate):
- `gpt2` → `gpt2-medium` or `gpt2-large` ✅
- `gpt2-medium` → `gpt2-large` ✅
- Small models in same family (e.g., `llama-7b` → `llama-13b`) ✅

Avoid (low acceptance rate):
- `distilgpt2` → `gpt2-medium` ❌ (distilled model has different distribution)
- Models from different families ❌

## Examples

See the [examples/](examples/) directory for complete examples:
- [examples/demo.py](examples/demo.py) - Complete working demo with benchmarking

Run the demo:
```bash
python examples/demo.py
```

## API Reference

`SpeculativeSampler` - Main class for speculative sampling.

**Methods:**
- `sample(input_ids, T)` - Generate T tokens using speculative sampling
- `auto_regressive_sample(input_ids, T)` - Generate T tokens using baseline sampling
- `sample_and_decode(text, T)` - End-to-end generation from text
- `encode(text)` - Convert text to token IDs
- `decode(token_ids)` - Convert token IDs to text

### Functions

- `speculative_sampling(...)` - Functional API for speculative sampling
- `auto_regressive_sampling(...)` - Functional API for baseline sampling

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- Transformers ≥ 4.30.0
- NumPy < 2.0

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{speculoos2026,
  title = {Speculoos: Efficient Speculative Sampling for Language Models},
  author = {Chakraborty, Bijin},
  year = {2026},
  url = {https://github.com/bijinc/speculoos}
}
```

Based on the paper:
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (Chen et. al, 2023)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and issues, please open an issue on [GitHub](https://github.com/bijinc/speculoos/issues).

