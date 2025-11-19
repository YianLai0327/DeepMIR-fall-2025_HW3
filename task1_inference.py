from main_g import test
import os
from tqdm import tqdm
import argparse

seeds = range(20)

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, help='the dictionary path', default='checkpoints/epoch_120.pkl')
parser.add_argument('--output_dir', type=str, help='the output directory', default='task1_results')

arg = parser.parse_args()
checkpoint_path = arg.ckpt
print(f"Starting inference for checkpoint: {checkpoint_path}")
temps = [0.8, 1.0, 1.2]

output_dir = arg.output_dir
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