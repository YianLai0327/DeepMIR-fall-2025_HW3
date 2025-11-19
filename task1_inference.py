from main_g import test
import os
from tqdm import tqdm

seeds = range(20)

# checkpoint_paths = ["checkpoints/epoch_120.pkl"]
checkpoint_path = "checkpoints/epoch_120.pkl"
print(f"Starting inference for checkpoint: {checkpoint_path}")
temps = [0.8, 1.0, 1.2]

output_dir = 'task1_results'
os.makedirs(output_dir, exist_ok=True)

for temp in temps:
    print(f"Generating results with temperature: {temp}")
    sub_dir = os.path.join(output_dir, f'temp_{temp}')
    os.makedirs(sub_dir, exist_ok=True)
    for seed in tqdm(seeds):
        output_path = os.path.join(sub_dir, f'{seed}.mid')
        test(
            n_target_bar=32,
            temperature=temp,
            topk=5,
            output_path=output_path,
            model_path=checkpoint_path,
        )
        print(f"Inference completed for seed {seed}, midi results saved to {output_path}, wav results saved to {output_path.replace('.mid', '.wav')}")