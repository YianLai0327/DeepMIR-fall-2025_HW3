import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataloader import DataLoader, Dataset
import numpy as np
import pickle
import utils
import os
import argparse
import warnings
import math
import torch.nn.functional as F
from pathlib import Path
from typing import List

# 引入 Hugging Face Transformers
from transformers import GPT2Config, GPT2LMHeadModel

warnings.filterwarnings('ignore')


## set the input length. must be same with the model config
X_LEN = 1024

def parse_opt():
    parser = argparse.ArgumentParser()
    # --- General ---
    parser.add_argument('--device', type=str, help='gpu device.', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dict_path', type=str, help='the dictionary path.', default='./basic_event_dictionary.pkl')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='train or test')

    # --- Training ---
    parser.add_argument('--data_path', type=str, default='Pop1K7/midi_analyzed', help='path to training data')
    parser.add_argument('--ckp_folder', type=str, default='checkpoints', help='path to save checkpoints')
    parser.add_argument('--continue_training_path', type=str, default='', help='path to a checkpoint to continue training')
    parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    
    # --- Testing ---
    parser.add_argument('--model_path', type=str, help='path to the model checkpoint for testing', default='./checkpoints/epoch_200.pkl')
    parser.add_argument('--output_path', type=str, default='./results/generated.mid', help='path to save the generated midi file')
    parser.add_argument('--n_target_bar', type=int, default=32, help='number of bars to generate')
    parser.add_argument('--temperature', type=float, default=1.2, help='temperature for sampling')
    parser.add_argument('--topk', type=int, default=5, help='top-k for sampling')
    
    # --- Model Architecture ---
    parser.add_argument('--n_layer', type=int, default=12, help='number of transformer layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_embd', type=int, default=512, help='embedding dimension')

    args = parser.parse_args()
    return args

opt = parse_opt()
event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
VOCAB_SIZE = len(event2word)


class NewsDataset(Dataset):
    def __init__(self, data_paths, prompt = ''):
        self.data_paths = data_paths
        self.x_len = X_LEN
        self.dictionary_path = opt.dict_path
        self.event2word, self.word2event = pickle.load(open(self.dictionary_path, 'rb'))
        self.parser = self.prepare_data(self.data_paths)
    
    def __len__(self):
        return len(self.parser)  
    
    def __getitem__(self, index):
        return self.parser[index]
    
    # def chord_extract(self, midi_path, max_time):
    #     note_items, _ = utils.read_items(midi_path)
    #     chords = utils.extract_chords(note_items)
    #     return chords
    
    def extract_events(self, input_path):
        note_items, tempo_items = utils.read_items(input_path)
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end if note_items else tempo_items[-1].start
        items = tempo_items + note_items
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups)
        return events
        
    def prepare_data(self, midi_paths):
        all_events = []
        for path in tqdm(midi_paths, desc="Processing MIDI files"):
            try:
                events = self.extract_events(path)
                all_events.append(events)
            except Exception as e:
                print(f"Error processing {path}: {e}")

        all_words = []
        for events in all_events:
            words = []
            for event in events:
                e = '{}_{}'.format(event.name, event.value)
                if e in self.event2word:
                    words.append(self.event2word[e])
                else:
                    if event.name == 'Note Velocity':
                        words.append(self.event2word['Note Velocity_31']) 
                    else:
                        print('Unknown event: {}'.format(e))
            all_words.append(words)
        
        segments = []
        for words in all_words:
            pairs = []
            for i in range(0, len(words) - self.x_len - 1, self.x_len // 2): 
                x = words[i:i+self.x_len]
                y = words[i+1:i+self.x_len+1]
                if len(x) == self.x_len and len(y) == self.x_len:
                    pairs.append([x, y])

            segments.extend(pairs)
            
        segments = np.array(segments)
        print(f"Total segments: {len(segments)}")
        return segments


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #################################################
        # 使用 Hugging Face GPT-2 模型
        #################################################
        # 1. 建立模型設定
        config = GPT2Config(
            vocab_size=VOCAB_SIZE,
            n_positions=X_LEN,        # 最大序列長度
            n_embd=opt.n_embd,        # 嵌入維度
            n_layer=opt.n_layer,      # Transformer 層數
            n_head=opt.n_head,        # 注意力頭數
            bos_token_id=0,           # 設定 dummy token id
            eos_token_id=0
        )
        
        # 2. 從設定檔初始化 GPT2LMHeadModel
        # 這個模型包含語言模型頭 (language modeling head)，適合生成任務
        self.gpt2 = GPT2LMHeadModel(config)

    def forward(self, x):
        #################################################
        # 定義前向傳播
        # GPT2LMHeadModel 會自動處理 causal mask
        #################################################
        # x shape: (batch_size, seq_len)
        outputs = self.gpt2(input_ids=x)
        return outputs.logits


def temperature_sampling(logits, temperature, topk):
    logits = logits / temperature
    topk_vals, topk_indices = torch.topk(logits, topk)
    probabilities = F.softmax(topk_vals, dim=-1)
    sampled_index_in_topk = torch.multinomial(probabilities, 1)
    sampled_token_index = topk_indices.gather(-1, sampled_index_in_topk)
    return sampled_token_index.item()
    
def test(n_target_bar=32, temperature=1.2, topk=5, output_path='', model_path='', prompt=False):
    try:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Directory '{output_dir}' is created")
    except Exception as e:
        print(f"Could not create output directory: {e}")
        return

    event2word, word2event = pickle.load(open(opt.dict_path, 'rb'))
    with torch.no_grad():
        if not os.path.exists(model_path):
            print(f"Model path does not exist: {model_path}")
            return
            
        checkpoint = torch.load(model_path, map_location=opt.device)
        model = Model().to(opt.device)
        # 載入 state_dict 時，因為模型現在是 self.gpt2，需要調整 key
        # Hugging Face 模型儲存的 state_dict 沒有 'gpt2.' 前綴
        # 我們需要從 checkpoint 中載入 gpt2 模型的權重
        state_dict = checkpoint['model']
        # 移除 'gpt2.' 前綴 (如果儲存時有)
        state_dict = {k.replace('gpt2.', ''): v for k, v in state_dict.items()}
        model.gpt2.load_state_dict(state_dict)
        model.eval()

        batch_size = 1

        if prompt:  
            print("Prompt-based generation is not fully implemented in this example.")
            words = [[event2word['Bar_None'], event2word['Position_1/16']]]
        else:  
            words = []
            for _ in range(batch_size):
                ws = [event2word['Bar_None']]
                tempo_classes = [v for k, v in event2word.items() if 'Tempo Class' in k]
                tempo_values = [v for k, v in event2word.items() if 'Tempo Value' in k]
                ws.append(event2word['Position_1/16'])
                ws.append(np.random.choice(tempo_classes))
                ws.append(np.random.choice(tempo_values))
                words.append(ws)

        generated_sequence = words[0]
        current_generated_bar = 0
        
        pbar = tqdm(total=n_target_bar, desc='Generating bars')
        while current_generated_bar < n_target_bar:
            input_tensor = torch.tensor([generated_sequence], dtype=torch.long, device=opt.device)
            if input_tensor.size(1) > X_LEN:
                input_tensor = input_tensor[:, -X_LEN:]
            
            output_logits = model(input_tensor)
            next_token_logits = output_logits[0, -1, :]
            
            next_token = temperature_sampling(
                logits=next_token_logits, 
                temperature=temperature,
                topk=topk)

            generated_sequence.append(next_token)

            if next_token == event2word['Bar_None']:
                current_generated_bar += 1
                pbar.update(1)
        
        pbar.close()
        
        print(f"\nGeneration complete. Writing to {output_path}")
        utils.write_midi(
            words=generated_sequence,
            word2event=word2event,
            output_path=output_path,
            prompt_path=None)

def train():
    epochs = opt.epochs
    DATA_DIR = Path(opt.data_path)
    train_list: List[Path] = list(DATA_DIR.glob("**/*.mid")) + list(DATA_DIR.glob("**/*.midi"))
    print('Number of training files found:', len(train_list))
    print(f"path: {train_list}")
    print('train list len =', len(train_list))

    train_dataset = NewsDataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    print('Dataloader is created')

    device = torch.device(opt.device)
    
    start_epoch = 1
    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    if opt.continue_training_path and os.path.isfile(opt.continue_training_path):
        print(f"Continuing training from {opt.continue_training_path}")
        checkpoint = torch.load(opt.continue_training_path, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")

    print('Model is created \nStart training')
    
    model.train()
    losses = []
    
    os.makedirs(opt.ckp_folder, exist_ok=True)
    
    for epoch in range(start_epoch, epochs+1):
        epoch_losses = []
        for i, data in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            x = data[:, 0, :].to(device).long()
            y = data[:, 1, :].to(device).long()
            
            optimizer.zero_grad()
            output_logit = model(x)
            
            loss = nn.CrossEntropyLoss()(output_logit.permute(0, 2, 1), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print('>>> Epoch: {}, Loss: {:.5f}'.format(epoch, avg_loss))
        if epoch % 10 == 0:
            print('Saving checkpoint...')
            torch.save({'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'loss': avg_loss,
                        }, os.path.join(opt.ckp_folder, f'epoch_{epoch:03d}.pkl'))
            
            np.save(os.path.join(opt.ckp_folder, 'training_loss.npy'), np.array(losses))

        if (epoch % 10 == 0) and avg_loss <= 1.2:
            print('Performing generation for evaluation...')
            # inference after certain epochs
            test(
                n_target_bar=32,
                temperature=1.2,
                topk=5,
                output_path=os.path.join(opt.ckp_folder, f'generated_epoch_{epoch:03d}.mid'),
                model_path=os.path.join(opt.ckp_folder, f'epoch_{epoch:03d}.pkl')
            )

def main():
    if opt.mode == 'train':
        train()
    elif opt.mode == 'test':
        test(
            n_target_bar=opt.n_target_bar,
            temperature=opt.temperature,
            topk=opt.topk,
            output_path=opt.output_path,
            model_path=opt.model_path
        )

if __name__ == '__main__':
    main()