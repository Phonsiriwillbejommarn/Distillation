# Knowledge Distillation: Qwen2.5-32B → Qwen2.5-3B

ถ่ายทอดความรู้จาก **Qwen2.5-32B-Instruct** (Teacher) สู่ **Qwen2.5-3B** (Student) ผ่าน 2 ขั้นตอน

```
Phase 1: SFT (Full Fine-Tuning)          Phase 2: Logit Distillation
┌─────────────────────────┐          ┌──────────────────────────────┐
│ Qwen2.5-3B (Base)       │          │ Teacher: Qwen2.5-32B (4-bit)│
│ + Opus Reasoning 3K     │  ──────► │ Student: SFT checkpoint      │
│ → sft_output/           │          │ + MATH 12.5K                 │
│ ~30-45 min (H100)       │          │ → distill_output/            │
└─────────────────────────┘          │ ~3-4 hrs (H100)              │
                                     └──────────────────────────────┘
```

---

## � ติดตั้ง

```bash
git clone https://github.com/Phonsiriwillbejommarn/Distillation.git
cd Distillation
pip install -r requirements.txt
```

---

## 🔑 ตั้งค่า API Keys

แก้ไขใน **ทั้ง `sft_qwen.py` และ `distill_qwen.py`**:

```python
MY_WANDB_KEY = "ใส่_wandb_key_จริง"   # https://wandb.ai/authorize
MY_HF_TOKEN  = "ใส่_hf_token_จริง"    # https://huggingface.co/settings/tokens (Write access)
```

---

## 🚀 Phase 1: SFT (Supervised Fine-Tuning)

สอน Qwen2.5-3B base ให้ทำ reasoning ด้วย [Opus Reasoning dataset](https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered)

```bash
python sft_qwen.py --config distill_config.yaml
```

| รายละเอียด | ค่า |
|-----------|-----|
| โมเดล | `Qwen/Qwen2.5-3B` (Base, Full Fine-Tuning) |
| Dataset | `nohurry/Opus-4.6-Reasoning-3000x-filtered` (3K ข้อ) |
| Max tokens | 8192 |
| Batch size | 4 × 4 = 16 (effective) |
| เวลา (H100) | ~30-45 นาที |
| Output | `./sft_output/` + HF: `Phonsiri/Qwen2.5-3B-SFT-Reasoning` |

---

## 🧠 Phase 2: Knowledge Distillation

ถ่ายทอดจาก Teacher 32B สู่ Student (SFT checkpoint) ด้วย [MATH dataset](https://huggingface.co/datasets/rasbt/math_full_minus_math500)

```bash
python distill_qwen.py \
    --student_model ./sft_output \
    --config distill_config.yaml
```

| รายละเอียด | ค่า |
|-----------|-----|
| Teacher | `Qwen/Qwen2.5-32B-Instruct` (4-bit quantized, frozen) |
| Student | `./sft_output` (Full Fine-Tuning) |
| Dataset | `rasbt/math_full_minus_math500` (12.5K ข้อ) |
| Loss | `α × KL(teacher ∥ student) × T² + (1-α) × CE` |
| Alpha | 0.5, Temperature: 2.0 |
| Checkpoint | เซฟทุก 100 steps → push ไป HF Hub |
| เวลา (H100) | ~3-4 ชั่วโมง |
| Output | `./distill_output/` + HF: `Phonsiri/Qwen2.5-3B-Math-Distilled` |

---

## ⏸️ Resume จาก Checkpoint

ถ้า GPU หลุดกลางคัน หรือต้องการรันต่อจากเมื่อวาน:

**วิธีที่ 1: ดึงจากโฟลเดอร์รันล่าสุด (ง่ายที่สุด)**
```bash
python distill_qwen.py \
    --student_model ./sft_output \
    --config distill_config.yaml \
    --resume_from_checkpoint auto
```
ระบบจะเข้าไปหาโฟลเดอร์ล่าสุดใน `./distill_output` และทำต่อให้อัตโนมัติ

**วิธีที่ 2: ระบุโฟลเดอร์เอง (กรณีโหลดมาจาก HuggingFace)**
ถ้าย้ายเครื่อง แนะนำให้โหลดโฟลเดอร์ checkpoint มาไว้ในเครื่อง แล้วระบุ path ตรงๆ:
```bash
python distill_qwen.py \
    --student_model ./sft_output \
    --config distill_config.yaml \
    --resume_from_checkpoint ./distill_output/last-checkpoint
```

🚨 *Checkpoint ทุกอันจะทยอยถูก Push ขึ้น Hugging Face Model Hub ของคุณอัตโนมัติหากตั้ง `push_to_hub: true`*

---

## 🔍 ใช้งานโมเดลหลังเทรน

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Phonsiri/Qwen2.5-3B-Math-Distilled")
tokenizer = AutoTokenizer.from_pretrained("Phonsiri/Qwen2.5-3B-Math-Distilled")

messages = [{"role": "user", "content": "What is the sum of 1+2+3+...+100?"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

โมเดลจะตอบในรูปแบบ:
```
<|im_start|>assistant
<think>
[กระบวนการคิด reasoning]
</think>

[คำตอบสุดท้าย]
<|im_end|>
```

---

## ⚙️ CLI Overrides

ทุกค่าใน `distill_config.yaml` สามารถ override ผ่าน CLI:

```bash
python distill_qwen.py --alpha 0.7 --temperature 3.0 --learning_rate 1e-5
python sft_qwen.py --max_seq_length 4096 --num_train_epochs 1
```

---

## 📁 โครงสร้างไฟล์

```
├── sft_qwen.py               # Phase 1: SFT
├── distill_qwen.py           # Phase 2: Logit Distillation
├── distill_config.yaml        # Configuration ทั้งหมด
├── generate_teacher_data.py   # (Optional) สร้าง teacher responses
├── requirements.txt           # Dependencies
└── README.md
```

---

## 💻 GPU Requirements

| GPU | SFT | Distillation | VRAM ใช้ |
|-----|-----|-------------|---------|
| **H100 85GB** | ~30 min | ~3-4 hrs | ~60 GB |
| A100 80GB | ~1.5 hrs | ~8 hrs | ~55 GB |
| A100 40GB | ~2 hrs | ~12 hrs | ~38 GB |

> VRAM ขั้นต่ำ: ~38 GB (Teacher 32B 4-bit + Student 3B Full)

---

## 📈 HuggingFace Hub Models

| Model | Repo |
|-------|------|
| SFT checkpoint | [Phonsiri/Qwen2.5-3B-Distilled](https://huggingface.co/Phonsiri/Qwen2.5-3B-Distilled) |
| Distilled (final) | [Phonsiri/Qwen2.5-3B-Math-Distilled](https://huggingface.co/Phonsiri/Qwen2.5-3B-Math-Distilled) |
