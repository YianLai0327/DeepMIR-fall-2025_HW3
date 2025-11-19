from main_g import continuation
import os
from tqdm import tqdm
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--ckpt', type=str, help='the dictionary path', default='checkpoints/epoch_120.pkl')
arg_parser.add_argument('--prompt_dir', type=str, help='the prompt midi directory', default='prompt_song')
arg_parser.add_argument('--output_dir', type=str, help='the output directory', default='task2_results')
arg = arg_parser.parse_args()

prompt_dir = arg.prompt_dir
temps = [0.8, 1.0, 1.2]

output_dir = arg.output_dir
os.makedirs(output_dir, exist_ok=True)

for temp in temps:
    print(f"Generating results with temperature: {temp}")
    sub_dir = os.path.join(output_dir, f'temp_{temp}')
    os.makedirs(sub_dir, exist_ok=True)
    for prompt in os.listdir(prompt_dir):
        if not prompt.endswith('.mid'):
            continue
        prompt_path = os.path.join(prompt_dir, prompt)
        output_path = os.path.join(sub_dir, prompt)
        continuation(
            prompt_midi_path=prompt_path,
            n_target_bar=24,
            temperature=temp,
            topk=5,
            output_path=output_path,
            model_path=arg.ckpt,
        )
        print(f"Inference completed for prompt {prompt}, midi results saved to {output_path}, wav results saved to {output_path.replace('.mid', '.wav')}")
        