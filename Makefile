# Makefile for Zen Guard (3B)

MODEL_NAME = zen-guard-3b
BASE_MODEL = Qwen/Qwen2.5-3B-Instruct
HF_REPO = zenlm/${MODEL_NAME}

.PHONY: all
all: train quantize upload

.PHONY: train
train:
	@echo "🎯 Training zen-guard..."
	@python train_zen_guard.py

.PHONY: quantize
quantize:
	@echo "🗜️ Creating GGUF quantizations..."
	@make gguf-q4 gguf-q5 gguf-q8

.PHONY: gguf-q4
gguf-q4:
	@../llama.cpp/build/bin/llama-quantize \
		gguf/${MODEL_NAME}-f16.gguf \
		gguf/${MODEL_NAME}-Q4_K_M.gguf Q4_K_M

.PHONY: mlx
mlx:
	@echo "🍎 Converting to MLX..."
	@mlx_lm.convert --hf-path finetuned --mlx-path mlx --quantize

.PHONY: test
test:
	@echo "🧪 Testing zen-guard..."
	@python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
		model = AutoModelForCausalLM.from_pretrained('finetuned'); \
		tokenizer = AutoTokenizer.from_pretrained('finetuned'); \
		print('Model loaded successfully')"

.PHONY: upload
upload:
	@echo "📤 Uploading to HuggingFace..."
	@huggingface-cli upload ${HF_REPO} . --repo-type model

.PHONY: clean
clean:
	@rm -rf finetuned/ gguf/ mlx/
