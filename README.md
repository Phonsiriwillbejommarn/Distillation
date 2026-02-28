# Knowledge Distillation: Qwen2.5-32B-Instruct → Qwen2.5-3B

ถ่ายทอดความรู้จาก **Qwen2.5-32B-Instruct** (Teacher) สู่ **Qwen2.5-3B** (Student)

## Pipeline

```
Phase 1: SFT (Full Fine-Tuning)     →  Phase 2: Logit Distillation (Full)
         Qwen2.5-3B base                       SFT checkpoint + Teacher 32B
         Opus Reasoning 3K                     MATH 12.5K
         ~30-45 min (H100)                     ~3-4 hrs (H100)
```

---

## 🚀 Quick Start

### 1. ติดตั้ง Dependencies

```bash
git clone https://github.com/Phonsiriwillbejommarn/Distillation.git
cd Distillation
pip install -r requirements.txt
```

### 2. ตั้งค่า API Keys

แก้ไขใน **ทั้ง 2 ไฟล์** (`sft_qwen.py` และ `distill_qwen.py`):

```python
MY_WANDB_KEY = "ใส่_wandb_key_จริง"
MY_HF_TOKEN = "ใส่_hf_token_จริง"
```

### 3. Phase 1: SFT (Supervised Fine-Tuning)

สอนโมเดลพื้นฐานให้ทำ reasoning ด้วย dataset `nohurry/Opus-4.6-Reasoning-3000x-filtered`

```bash
python sft_qwen.py --config distill_config.yaml
```

โมเดล SFT จะถูกเซฟที่ `./sft_output/` และ push ไปที่ `Phonsiri/Qwen2.5-3B-SFT-Reasoning`

### 4. Phase 2: Knowledge Distillation

ถ่ายทอดความรู้จาก Teacher (32B) สู่ Student (SFT checkpoint) ด้วย dataset `rasbt/math_full_minus_math500`

```bash
python distill_qwen.py \
    --student_model ./sft_output \
    --config distill_config.yaml
```

โมเดลสุดท้ายจะถูกเซฟที่ `./distill_output/` และ push ไปที่ `Phonsiri/Qwen2.5-3B-Distilled`

---

## ⏸️ Resume จาก Checkpoint

ถ้าเทรนหยุดกลางคัน (GPU หมดเวลา, error, etc.):

```bash
# Auto-resume จาก checkpoint ล่าสุด
python distill_qwen.py \
    --student_model ./sft_output \
    --config distill_config.yaml \
    --resume_from_checkpoint auto

# หรือระบุ checkpoint ที่ต้องการ
python distill_qwen.py \
    --student_model ./sft_output \
    --config distill_config.yaml \
    --resume_from_checkpoint ./distill_output/checkpoint-500
```

> Checkpoint จะถูกเซฟทุก **100 steps** (เก็บไว้สูงสุด 5 ตัว)

---

## ⚙️ Configuration

แก้ไขได้ที่ `distill_config.yaml`:

| Parameter | Default | คำอธิบาย |
|-----------|---------|---------|
| `teacher_model` | `Qwen/Qwen2.5-32B-Instruct` | โมเดล Teacher |
| `student_model` | `Qwen/Qwen2.5-3B` | โมเดล Student |
| `alpha` | `0.5` | น้ำหนัก KL-div (0=SFT, 1=Distill เต็มที่) |
| `temperature` | `2.0` | อุณหภูมิ softmax (สูง=กระจายมากขึ้น) |
| `max_seq_length` | `8192` | ความยาวโทเคนสูงสุด |
| `per_device_train_batch_size` | `4` | Batch size ต่อ GPU |
| `gradient_accumulation_steps` | `4` | Effective batch = 4×4 = 16 |
| `learning_rate` | `2e-5` | อัตราการเรียนรู้ |
| `num_train_epochs` | `3` | จำนวน epochs |
| `save_steps` | `100` | เซฟ checkpoint ทุกกี่ steps |
| `teacher_load_in_4bit` | `true` | โหลด teacher แบบ 4-bit (ประหยัด VRAM) |

ทุกค่าสามารถ override ผ่าน CLI ได้:

```bash
python distill_qwen.py --alpha 0.7 --temperature 3.0 --learning_rate 1e-5
```

---

## 📊 Loss Function

```
L = α × KL(teacher_soft || student_soft) × T² + (1-α) × CE(labels, student)
```

- **KL-divergence**: Student เรียนรู้การกระจายความน่าจะเป็นของ teacher
- **Cross-Entropy**: Student เรียนรู้จาก ground truth labels
- **T² scaling**: ชดเชยการ scale ของ gradients จาก temperature

---

## 🔍 วิธีใช้งานโมเดลหลังเทรน

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Phonsiri/Qwen2.5-3B-Distilled")
tokenizer = AutoTokenizer.from_pretrained("Phonsiri/Qwen2.5-3B-Distilled")

messages = [{"role": "user", "content": "What is the sum of 1+2+3+...+100?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

Output จะมี format:
```
<|im_start|>assistant
<think>
[กระบวนการคิด]
</think>

[คำตอบสุดท้าย]
<|im_end|>
```

---

## 📁 โครงสร้างไฟล์

```
.
├── distill_config.yaml       # Configuration ทั้งหมด
├── sft_qwen.py               # Phase 1: SFT (Full Fine-Tuning)
├── distill_qwen.py           # Phase 2: Logit Distillation (Full)
├── generate_teacher_data.py  # (Optional) สร้าง teacher responses
├── requirements.txt          # Dependencies
└── README.md                 # เอกสารนี้
```

---

## 💻 GPU Requirements

| GPU | SFT | Distillation | รวม |
|-----|-----|-------------|-----|
| H100 85GB | ~30 min | ~3-4 hrs | ~4-5 hrs |
| A100 80GB | ~1.5 hrs | ~8 hrs | ~10 hrs |
| A100 40GB | ~2 hrs | ~12 hrs | ~14 hrs |
| RTX 4090 24GB | ~3 hrs | ~18 hrs | ~21 hrs |

> VRAM ขั้นต่ำ: ~38 GB (Teacher 32B 4-bit + Student 3B Full)

---

## 📈 Multi-GPU

```bash
accelerate launch --num_processes 2 distill_qwen.py --config distill_config.yaml
```
