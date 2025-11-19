from main_g import continuation
import os
from tqdm import tqdm

prompt_dir = "prompt_song"
temps = [0.8, 1.0, 1.2]

output_dir = 'task2_results'
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
            model_path="checkpoints/epoch_120.pkl",
        )
        print(f"Inference completed for prompt {prompt}, midi results saved to {output_path}, wav results saved to {output_path.replace('.mid', '.wav')}")
        