# DeepMIR HW3

## Checkpoint Installation

Download the model checkpoint:

```bash
gdown 1biwPpTqC_VGjqKXzK5ItJ-Gj-13gyl8t -O ckpt.pkl
```

This will save the checkpoint as `ckpt.pkl`.

---

## Inference

### Task 1

Run inference for Task 1:

```bash
python3 task1_inference.py \
    --ckpt /path/to/checkpoint/ \
    --output_dir /path/to/output_dir/
```

### Task 2

Run inference for Task 2:

```bash
python3 task2_inference.py \
    --ckpt /path/to/checkpoint/ \
    --prompt_dir /path/to/prompts/ \
    --output_dir /path/to/output_dir/
```
