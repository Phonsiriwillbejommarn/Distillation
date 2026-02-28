# Knowledge Distillation: Qwen2.5-32B-Instruct → Qwen2.5-3B

ถ่ายทอดความรู้ (Knowledge Distillation) จาก **Qwen2.5-32B-Instruct** (Teacher) สู่ **Qwen2.5-3B** (Student) ด้วย 2 แนวทาง:

| Approach | Script | Memory | Quality |
|----------|--------|--------|---------|
| **Logit-based** | `distill_qwen.py` | ~24 GB VRAM | ⭐⭐⭐ Higher |
| **Response-based** | `generate_teacher_data.py` | ~18 GB VRAM | ⭐⭐ Good |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Option A: Logit-Based Distillation (Recommended)

The student learns to match the teacher's full probability distribution via KL-divergence loss.

```bash
# Default config
python distill_qwen.py --config distill_config.yaml

# Custom settings
python distill_qwen.py \
    --alpha 0.7 \
    --temperature 3.0 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1

# Quick test (1 step)
python distill_qwen.py --max_steps 1
```

### 3. Option B: Response-Based Distillation

Step 1: Generate teacher responses offline:

```bash
python generate_teacher_data.py --config distill_config.yaml

# With sample limit (for testing)
python generate_teacher_data.py --max_samples 100
```

Step 2: Fine-tune the student on teacher data:

```bash
python distill_qwen.py \
    --dataset_name ./teacher_responses.jsonl \
    --alpha 0.0
```

> Setting `--alpha 0.0` disables KL-divergence loss → pure SFT on teacher outputs.

---

## Configuration

All settings are in `distill_config.yaml`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.5` | Weight for KL-div loss (0 = pure SFT, 1 = pure distillation) |
| `temperature` | `2.0` | Softmax temperature (higher = softer distributions) |
| `use_lora` | `true` | Use LoRA adapters for student |
| `lora_r` | `64` | LoRA rank |
| `teacher_load_in_4bit` | `true` | Quantize teacher to 4-bit |
| `per_device_train_batch_size` | `2` | Batch size per GPU |
| `gradient_accumulation_steps` | `8` | Effective batch = batch_size × grad_accum |

---

## Loss Function

```
L_total = α × KL(teacher_soft || student_soft) × T² + (1 - α) × CE(labels, student)
```

Where:
- `teacher_soft = softmax(teacher_logits / T)`
- `student_soft = softmax(student_logits / T)`
- `T` = temperature (default: 2.0)
- `α` = distillation weight (default: 0.5)

---

## Custom Dataset

To use your own dataset, set `dataset_name` to a local JSONL file:

```yaml
# distill_config.yaml
dataset_name: "./my_data.jsonl"
```

Expected JSONL format (Alpaca-style):
```json
{"instruction": "Translate to Thai", "input": "Hello", "output": "สวัสดี"}
```

Or pre-formatted text:
```json
{"text": "### Instruction:\nTranslate to Thai\n\n### Input:\nHello\n\n### Response:\nสวัสดี"}
```

---

## Multi-GPU

Both scripts support multi-GPU via `accelerate`:

```bash
accelerate launch --num_processes 4 distill_qwen.py --config distill_config.yaml
```

---

## Output

After training, the model is saved to `./distill_output/` (configurable). To load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# If LoRA was used
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
model = PeftModel.from_pretrained(base_model, "./distill_output")

# Or merge LoRA into base
model = model.merge_and_unload()
model.save_pretrained("./qwen2.5-3b-distilled-merged")
```

To push to HuggingFace Hub:
```bash
python distill_qwen.py --push_to_hub --hub_model_id your-name/qwen2.5-3b-distilled
```
