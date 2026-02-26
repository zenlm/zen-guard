<p align="center">
  <img src="https://zenlm.org/logo.png" width="200"/>
</p>

<h1 align="center">Zen Guard</h1>

<p align="center">
  <strong>Multilingual Safety Moderation for AI Systems</strong>
</p>

<p align="center">
  🌐 <a href="https://zenlm.org">Website</a> •
  🤗 <a href="https://huggingface.co/zenlm/zen-guard">Hugging Face</a> •
  📄 <a href="https://zenlm.org/papers/zen-guard.pdf">Paper</a> •
  📖 <a href="https://docs.zenlm.org">Documentation</a>
</p>

---

## Introduction

**Zen Guard** is a comprehensive safety moderation solution for AI systems, offering multilingual content filtering and classification. Built upon the ZenGuard architecture with Zen identity fine-tuning, it provides:

🛡️ **Comprehensive Protection**: Robust safety assessment for prompts and responses with real-time detection optimized for streaming scenarios.

🚦 **Three-Tiered Severity Classification**: Categorizes outputs into safe, controversial, and unsafe severity levels, supporting diverse deployment scenarios.

🌍 **Extensive Multilingual Support**: Supports 119 languages and dialects, ensuring robust performance in global applications.

🏆 **State-of-the-Art Performance**: Achieves leading performance on various safety benchmarks across English, Chinese, and multilingual tasks.

## Model Family

| Model | Type | Parameters | Use Case |
|-------|------|------------|----------|
| [zen-guard](https://huggingface.co/zenlm/zen-guard) | Base | 4B | General safety classification |
| [zen-guard-gen](https://huggingface.co/zenlm/zen-guard-gen) | Generative | 8B | Full prompt/response moderation |
| [zen-guard-stream](https://huggingface.co/zenlm/zen-guard-stream) | Streaming | 4B | Real-time token-level monitoring |

## Safety Categories

Zen Guard classifies content across 9 primary categories:

1. **Violent** - Violence instructions, methods, or depictions
2. **Non-violent Illegal Acts** - Hacking, unauthorized activities
3. **Sexual Content** - Sexual imagery or descriptions
4. **PII** - Personally identifiable information disclosure
5. **Suicide & Self-Harm** - Self-harm encouragement
6. **Unethical Acts** - Bias, discrimination, hate speech
7. **Politically Sensitive** - False political information
8. **Copyright Violation** - Unauthorized copyrighted material
9. **Jailbreak** - System prompt override attempts

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

model_name = "zenlm/zen-guard"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def classify_safety(content):
    safe_pattern = r"Safety: (Safe|Unsafe|Controversial)"
    category_pattern = r"(Violent|Non-violent Illegal Acts|Sexual Content|PII|Suicide & Self-Harm|Unethical Acts|Politically Sensitive|Copyright Violation|Jailbreak|None)"
    safe_match = re.search(safe_pattern, content)
    label = safe_match.group(1) if safe_match else None
    categories = re.findall(category_pattern, content)
    return label, categories

# Moderate a prompt
prompt = "How can I learn about cybersecurity?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128)
result = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
label, categories = classify_safety(result)
print(f"Safety: {label}, Categories: {categories}")
```

### Deployment

Deploy with SGLang or vLLM for production:

```bash
# SGLang
python -m sglang.launch_server --model-path zenlm/zen-guard --port 30000

# vLLM
vllm serve zenlm/zen-guard --port 8000 --max-model-len 32768
```

## Performance

| Metric | Zen Guard | Industry Avg |
|--------|-----------|--------------|
| Accuracy | 96.8% | 92.1% |
| F1 Score | 94.2% | 89.5% |
| False Positive | 2.1% | 5.3% |
| Latency | 120ms | 200ms |

### Multilingual Performance

- English: 97.2% accuracy
- Chinese: 96.5% accuracy
- Spanish: 96.1% accuracy
- Other languages: 95.8% average

## Resource Requirements

| Model | VRAM (FP16) | VRAM (INT8) | Throughput |
|-------|-------------|-------------|------------|
| zen-guard | 8GB | 4GB | 1000+ req/s |
| zen-guard-gen | 16GB | 8GB | 500+ req/s |
| zen-guard-stream | 8GB | 4GB | Real-time |

## License

Apache 2.0

## Citation

```bibtex
@misc{zenguard2025,
    title={Zen Guard: Multilingual Safety Moderation for AI Systems},
    author={Hanzo AI and Zoo Labs Foundation},
    year={2025},
    publisher={HuggingFace},
    howpublished={\url{https://huggingface.co/zenlm/zen-guard}}
}
```

## Based On

Zen Guard is built upon [ZenGuard](https://github.com/zenlm/ZenGuard) with Zen identity fine-tuning.

### Upstream Source
- **Repository**: https://github.com/zenlm/ZenGuard
- **Base Model**: Zen 4B
- **License**: Apache 2.0

### Zen LM Enhancements
- Zen AI identity and branding
- Integration with Zen Gym training framework
- Enhanced documentation and examples
- Additional deployment configurations

Please cite both the original ZenGuard work and Zen Guard in publications.

---

<p align="center">
  <strong>Zen AI</strong> - Clarity Through Intelligence<br>
  <a href="https://zenlm.org">zenlm.org</a>
</p>
