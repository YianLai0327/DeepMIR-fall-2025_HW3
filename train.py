# train.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import TransfoXLConfig, TransfoXLLMHeadModel, GPT2Config, GPT2LMHeadModel
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from miditok import REMI
import shutil
import subprocess

# 導入你的 dataloader
from dataloader import create_dataloaders

# def generate_music(
#     model,
#     tokenizer,
#     num_bars: int = 32,
#     temperature: float = 1.0,
#     top_k: int = 50,
#     top_p: float = 0.95,
#     device: str = "cuda",
#     seed: int = None,
#     vocab_size: int = 3000,  # 添加 vocab_size 参数
# ):
#     """
#     生成 unconditional music sequence
#     """
#     model.eval()
    
#     if seed is not None:
#         torch.manual_seed(seed)
    
#     # 估算需要的 token 数量
#     tokens_per_bar = 81
#     max_length = num_bars * tokens_per_bar
    
#     # ===== 关键修正：确保起始 token 有效 =====
#     # 方法 1: 使用一个安全的起始 token
#     # 避免使用 PAD (0)，选择一个在训练数据中常见的 token
    
#     # 从训练数据的统计中选择最常见的起始 token
#     # 通常 REMI 的序列以 Bar 或 Position token 开始
#     # 这里我们简单地使用一个中间值，避免边界情况
#     start_token_id = 1  # 或者可以是 tokenizer 的 BOS token
    
#     # 确保起始 token 在有效范围内
#     start_token_id = max(1, min(start_token_id, vocab_size - 1))
    
#     generated_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    
#     print(f"  Starting generation from token {start_token_id}")
#     print(f"  Target length: ~{max_length} tokens ({num_bars} bars)")
#     print(f"  Vocab size: {vocab_size}")
    
#     # 逐步生成
#     with torch.no_grad():
#         for step in range(max_length - 1):
#             # Forward pass
#             outputs = model(input_ids=generated_ids)
#             next_token_logits = outputs.logits[:, -1, :].clone()  # (1, vocab_size)
            
#             # ===== 关键修正：限制 logits 到有效的 vocab 范围 =====
#             # 将超出 vocab_size 的 logits 设为 -inf
#             if next_token_logits.shape[-1] > vocab_size:
#                 next_token_logits[:, vocab_size:] = float('-inf')
            
#             # 应用温度
#             next_token_logits = next_token_logits / temperature
            
#             # Top-k 过滤
#             if top_k > 0:
#                 top_k_actual = min(top_k, next_token_logits.shape[-1])
#                 indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_actual)[0][..., -1, None]
#                 next_token_logits[indices_to_remove] = float('-inf')
            
#             # Top-p (nucleus) 过滤
#             if top_p < 1.0:
#                 sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
#                 cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
#                 sorted_indices_to_remove = cumulative_probs > top_p
#                 sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
#                 sorted_indices_to_remove[..., 0] = 0
                
#                 indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
#                 next_token_logits[indices_to_remove] = float('-inf')
            
#             # ===== 关键修正：检查是否有有效的 tokens =====
#             # 如果所有 logits 都是 -inf，回退到均匀分布
#             if torch.all(torch.isinf(next_token_logits)):
#                 print(f"  Warning: All logits filtered at step {step}, using uniform distribution")
#                 next_token_logits = torch.zeros_like(next_token_logits)
            
#             # 采样
#             probs = torch.softmax(next_token_logits, dim=-1)
            
#             # ===== 关键修正：确保概率有效 =====
#             if torch.any(torch.isnan(probs)) or torch.all(probs == 0):
#                 print(f"  Warning: Invalid probabilities at step {step}, using uniform")
#                 probs = torch.ones_like(probs) / probs.shape[-1]
            
#             next_token = torch.multinomial(probs, num_samples=1)
            
#             # ===== 关键修正：验证生成的 token =====
#             next_token_value = next_token.item()
#             if next_token_value < 0 or next_token_value >= vocab_size:
#                 print(f"  Warning: Generated invalid token {next_token_value}, clipping to valid range")
#                 next_token_value = max(1, min(next_token_value, vocab_size - 1))
#                 next_token = torch.tensor([[next_token_value]], dtype=torch.long, device=device)
            
#             # 添加到序列
#             generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
#             # 进度显示（每 100 steps）
#             if (step + 1) % 100 == 0:
#                 print(f"  Generated {step + 1}/{max_length} tokens...")
            
#             # 检查是否达到目标长度
#             if generated_ids.shape[1] >= max_length:
#                 break
    
#     print(f"  ✓ Generation completed: {generated_ids.shape[1]} tokens")
#     return generated_ids[0].cpu().tolist()

def generate_music(
    model,
    tokenizer,
    num_bars: int = 32,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda",
    seed: int = None,
    vocab_size: int = 3000,
):
    """
    生成 unconditional music sequence
    - 總長度對齊「num_bars」，但每次丟進 model 的 context 長度 <= model.config.n_positions
    """
    model.eval()
    
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
    
    # 粗估每小節 token 數
    tokens_per_bar = 81              # 你原本的估計，之後可以根據 dataset 平均再修
    max_length = num_bars * tokens_per_bar

    # 取得 model 能吃的最大 context 長度（GPT2 用 n_positions）
    max_context = getattr(model.config, "n_positions", None)
    if max_context is None:
        max_context = getattr(model.config, "max_position_embeddings", 512)
    if max_context is None:
        max_context = 512  # fallback

    print(f"  Target bars: {num_bars}")
    print(f"  Approx target length (tokens): {max_length}")
    print(f"  Model max context length: {max_context}")
    print(f"  Vocab size: {vocab_size}")

    # 起始 token（不要用 0 比較安全）
    start_token_id = 1
    start_token_id = max(1, min(start_token_id, vocab_size - 1))
    
    generated_ids = torch.tensor([[start_token_id]], dtype=torch.long, device=device)
    print(f"  Starting generation from token {start_token_id}")
    
    with torch.no_grad():
        for step in range(max_length - 1):
            # 🔥 sliding window：只拿最後 max_context 個 token 丟進 model
            input_ids = generated_ids[:, -max_context:]
            
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[:, -1, :].clone()  # (1, vocab)

            # 確保不會選到超出 vocab 的 id
            if next_token_logits.shape[-1] > vocab_size:
                next_token_logits[:, vocab_size:] = float('-inf')
            
            # 溫度
            next_token_logits = next_token_logits / temperature
            
            # Top-k
            if top_k > 0:
                top_k_actual = min(top_k, next_token_logits.shape[-1])
                kth_vals = torch.topk(next_token_logits, top_k_actual)[0][..., -1, None]
                indices_to_remove = next_token_logits < kth_vals
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 若全被過濾掉，回退成 uniform
            if torch.all(torch.isinf(next_token_logits)):
                print(f"  Warning: All logits filtered at step {step}, using uniform distribution")
                next_token_logits = torch.zeros_like(next_token_logits)
            
            probs = torch.softmax(next_token_logits, dim=-1)
            if torch.any(torch.isnan(probs)) or torch.all(probs == 0):
                print(f"  Warning: Invalid probabilities at step {step}, using uniform")
                probs = torch.ones_like(probs) / probs.shape[-1]
            
            next_token = torch.multinomial(probs, num_samples=1)

            # 保險：把 token id 壓回合法範圍
            next_token_value = next_token.item()
            if next_token_value < 0 or next_token_value >= vocab_size:
                print(f"  Warning: Generated invalid token {next_token_value}, clipping to valid range")
                next_token_value = max(1, min(next_token_value, vocab_size - 1))
                next_token = torch.tensor([[next_token_value]], dtype=torch.long, device=device)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if (step + 1) % 100 == 0:
                print(f"  Generated {step + 1}/{max_length - 1} tokens...")
            
            if generated_ids.shape[1] >= max_length:
                break
    
    print(f"  ✓ Generation completed: {generated_ids.shape[1]} tokens "
          f"(approx {generated_ids.shape[1]/tokens_per_bar:.1f} bars)")
    return generated_ids[0].cpu().tolist()

def save_generated_midi(
    token_ids,
    tokenizer: REMI,
    output_path: str,
):
    """
    将生成的 tokens 转换为 MIDI 并保存
    """
    try:
        # ===== 关键修正：正确使用 REMI tokenizer =====
        # REMI 的 decode 方法需要 token IDs
        
        # 方法 1: 使用 ids_to_tokens + tokens_to_midi
        # tokens = tokenizer.ids_to_tokens(token_ids)
        generated_midi = tokenizer.decode([token_ids])
        # generated_midi = (token_ids)
        
        # 保存 MIDI
        generated_midi.dump_midi(output_path)
        print(f"  ✓ Saved MIDI to: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to save MIDI: {e}")
        print(f"  Token IDs range: [{min(token_ids)}, {max(token_ids)}]")
        print(f"  Number of tokens: {len(token_ids)}")
        
        # 尝试保存 token IDs 以供调试
        try:
            import json
            debug_path = output_path.replace('.mid', '_tokens.json')
            with open(debug_path, 'w') as f:
                json.dump({
                    'token_ids': token_ids,  # 只保存前 100 个
                    'total_tokens': len(token_ids),
                    'min_id': min(token_ids),
                    'max_id': max(token_ids),
                }, f, indent=2)
            print(f"  ✓ Saved debug tokens to: {debug_path}")
        except:
            pass
        
        return False


def run_inference_test(
    model,
    tokenizer: REMI,
    epoch: int,
    output_dir: str,
    vocab_size: int,  # 添加 vocab_size 参数
    device: str = "cuda",
):
    """
    运行 inference 测试
    """
    print(f"\n{'='*60}")
    print(f"Running inference test at epoch {epoch}")
    print(f"{'='*60}")
    
    epoch_dir = Path(output_dir) / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # 测试不同的采样配置
    configs = [
        {"name": "greedy", "temperature": 0.8, "top_k": 50, "top_p": 0.95},
        # {"name": "diverse", "temperature": 1.2, "top_k": 100, "top_p": 0.9},
        {"name": "conservative", "temperature": 0.6, "top_k": 30, "top_p": 0.95},
    ]
    
    success_count = 0
    
    for config in configs:
        midi_saved = False
        print(f"\n{'='*50}")
        print(f"Config: {config['name']}")
        print(f"  Temperature: {config['temperature']}")
        print(f"  Top-k: {config['top_k']}")
        print(f"  Top-p: {config['top_p']}")
        print(f"{'='*50}")
        
        try:
            # 生成
            generated_ids = generate_music(
                model=model,
                tokenizer=tokenizer,
                num_bars=32,
                temperature=config['temperature'],
                top_k=config['top_k'],
                top_p=config['top_p'],
                device=device,
                seed=42 + configs.index(config),  # 不同的种子
                vocab_size=vocab_size,  # 传递 vocab_size
            )
            
            print(f"\n  Generated token statistics:")
            print(f"    Total: {len(generated_ids)}")
            print(f"    Range: [{min(generated_ids)}, {max(generated_ids)}]")
            print(f"    First 30: {generated_ids[:30]}")
            
            # 保存 MIDI
            output_path = epoch_dir / f"{config['name']}.mid"
            if save_generated_midi(generated_ids, tokenizer, str(output_path)):
                success_count += 1
                midi_saved = True
                
        except Exception as e:
            print(f"\n  ✗ Generation failed: {e}")
            import traceback
            traceback.print_exc()

        if midi_saved and output_path.exists():
            # covert midi to wav using fluidsynth
            try:
                wav_output_path = epoch_dir / f"{config['name']}.wav"
                midi_to_wav_with_fluidsynth(
                    midi_path=str(output_path),
                    wav_path=str(wav_output_path),
                    sound_font="/usr/share/sounds/sf2/FluidR3_GM.sf2",
                    sample_rate=44100,
                )
            except Exception as e:
                print(f"\n  ✗ WAV conversion failed: {e}")
                import traceback
                traceback.print_exc()

        else:
            print(f"\n  ✗ MIDI file not found, skipping WAV conversion.")

        # show progress
        print(f"  ✓ Finished generation for config: {config['name']}")

        
    print(f"\n{'='*60}")
    print(f"✓ Inference test completed: {success_count}/{len(configs)} successful")
    print(f"  Files saved to: {epoch_dir}")
    print(f"{'='*60}")

def midi_to_wav_with_fluidsynth(
    midi_path: str,
    wav_path: str,
    sound_font: str,
    sample_rate: int = 44100,
):
    """
    使用 Fluidsynth 將 MIDI 轉成 WAV。

    需要：
    - 系統已安裝 `fluidsynth` 指令
    - 有可用的 .sf2 soundfont 檔案
    """
    if shutil.which("fluidsynth") is None:
        raise RuntimeError(
            "fluidsynth not found in PATH，請先安裝 Fluidsynth，"
            "例如: sudo apt-get install fluidsynth"
        )

    midi_path = str(midi_path)
    wav_path = str(wav_path)

    cmd = [
        "fluidsynth",
        "-ni", sound_font,   # soundfont
        midi_path,           # midi input
        "-F", wav_path,      # output wav file
        "-r", str(sample_rate),
    ]

    print(f"  → Converting MIDI to WAV with Fluidsynth:")
    print(f"    Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  ✓ WAV saved to: {wav_path}")

def train_model(
    corpus_path: str = "tokenizer/remi_tokenized_corpus.pkl",
    tokenizer_path: str = "tokenizer/remi_tokenizer.json",
    batch_size: int = 32,  # 根據你的 GPU 調整
    seq_len: int = 512,
    num_epochs: int = 100,
    learning_rate: float = 2.5e-4,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
):
    # 建立 checkpoint 目錄
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    # ===== 1. 載入資料 =====
    print("="*60)
    print("Loading data...")
    print("="*60)
    
    train_loader, vocab_size = create_dataloaders(
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        batch_size=batch_size,
        seq_len=seq_len,
        overlap=64,
        # num_workers=4,  # 確認可以用後再調整
        num_workers=12,     # CPU 核心多就開大
        # train_ratio=0.95,
    )

    tokenizer = REMI(params=tokenizer_path)
    
    # ===== 2. 建立模型 =====
    print("\n" + "="*60)
    print("Building model...")
    print("="*60)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=seq_len,      # 最大序列長度
        n_embd=512,               # embedding dimension
        n_layer=12,               # number of layers
        n_head=8,                 # attention heads
        n_inner=2048,             # FFN dimension
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )

    model = GPT2LMHeadModel(config).to(device)
    # config = TransfoXLConfig()
    # model = TransfoXLLMHeadModel(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model has {total_params} total parameters, "
            f"{trainable_params} trainable parameters.")
    
    print(f"Model Configuration:")
    print(config)
    
    # ===== 3. 優化器和排程器 =====
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-5,
    )
    
    # ===== 4. 訓練歷史記錄 =====
    history = {
        'train_loss': [],
        # 'val_loss': [],
        'learning_rate': [],
    }
    
    best_val_loss = float('inf')

    LOSS_THRESHOLDS = [6.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.2, 1.0]
    generated_at_loss = set()
    
    # ===== 5. 訓練循環 =====
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                labels=target_ids,
            )
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
            # 更新進度條
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{train_loss/train_steps:.4f}'
            })
        
        avg_train_loss = train_loss / train_steps
        
        # # --- Validation ---
        # model.eval()
        # val_loss = 0.0
        # val_steps = 0
        
        # with torch.no_grad():
        #     for input_ids, target_ids in val_loader:
        #         input_ids = input_ids.to(device)
        #         target_ids = target_ids.to(device)
                
        #         outputs = model(
        #             input_ids=input_ids,
        #             labels=target_ids,
        #         )
        #         val_loss += outputs.loss.item()
        #         val_steps += 1
        
        # avg_val_loss = val_loss / val_steps
        
        # 更新學習率
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # 記錄歷史
        history['train_loss'].append(avg_train_loss)
        # history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        
        # 輸出結果
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        # print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR:         {current_lr:.2e}")
        print(f"{'='*60}\n")

        if (epoch % 10 == 0) and (epoch >= 60) and (avg_train_loss <= 1.2):
        # if True:
            # inference when reaching certain loss thresholds
            print(f"✓ Running inference test at epoch {epoch} with train loss {avg_train_loss:.4f}")
            run_inference_test(
                model=model,
                tokenizer=tokenizer,
                epoch=epoch,
                output_dir=checkpoint_dir,
                device=device,
                vocab_size=vocab_size,
            )
            for thr in LOSS_THRESHOLDS:
                if avg_train_loss < thr:
                    generated_at_loss.add(thr)

        
        # # 儲存最佳模型
        # if avg_val_loss < best_val_loss:
        #     best_val_loss = avg_val_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'train_loss': avg_train_loss,
        #         'val_loss': avg_val_loss,
        #         'config': config.to_dict(),
        #     }, f'{checkpoint_dir}/best_model.pt')
        #     print(f"✓ Saved best model (val_loss: {avg_val_loss:.4f})")
        
        # 定期儲存 checkpoint
        if (epoch) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                # 'val_loss': avg_val_loss,
                'config': config.to_dict(),
            }, f'{checkpoint_dir}/checkpoint_epoch_{epoch}_loss_{train_loss}.pt')
            print(f"✓ Saved checkpoint at epoch {epoch}")
                
        # 儲存訓練歷史
        with open(f'{checkpoint_dir}/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    # ===== 6. 繪製訓練曲線 =====
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    # plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rate'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{checkpoint_dir}/training_curves.png', dpi=150)
    print(f"\n✓ Training curves saved to {checkpoint_dir}/training_curves.png")
    
    return model, history


if __name__ == "__main__":
    # 檢查 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 開始訓練
    model, history = train_model(
        batch_size=32,  # 根據 GPU 記憶體調整
        seq_len=512,
        num_epochs=120,
        device=device,
    )
    
    print("\n" + "="*60)
    print("✓ Training completed!")
    print("="*60)