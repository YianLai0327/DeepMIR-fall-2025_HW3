# import pickle
# from pathlib import Path
# from typing import List
# from miditok import TokenizerConfig, REMI, CPWord

# # --- 1. 從 HW3.pdf (p.18) 載入建議的參數 ---
# # 
# # TOKENIZER_PARAMS = {
# #     "pitch_range": (21, 109),  # [cite: 325]
# #     "beat_res": {(0, 4): 8, (4, 12): 4},  # [cite: 326]
# #     "nb_velocities": 32,  # [cite: 327] (MidiTok uses nb_velocities)
# #     "additional_tokens": {
# #         'Chord': True,    # [cite: 328]
# #         'Rest': True,     # [cite: 328]
# #         'Tempo': True,    # [cite: 329]
# #         'rest_range': (2, 8),  # [cite: 330] (beats)
# #         'num_tempos': 32,      # [cite: 332]
# #         'tempo_range': (40, 250), # [cite: 332]
# #         'Program': False,  # [cite: 332]
# #     }
# # }
# tokenizer_config = TokenizerConfig(
#     pitch_range=(21, 109),  # [cite: 325]
#     beat_res={(0, 4): 8, (4, 12): 4},  # [cite: 326]
#     num_velocities=32,  # [cite: 327] (MidiTok uses nb_velocities)
#     use_chords=True,
#     use_rests=True,
#     use_tempos=True,
#     use_programs=False,
#     use_time_signatures = False,
# )

# # --- 2. choose your Tokenizer ---
# # tokenizer = REMI(tokenizer_config=tokenizer_config)

# tokenizer = CPWord(tokenizer_config=tokenizer_config)

# print(f"--- Tokenizer: {tokenizer.__class__.__name__} ---")

# DATA_DIR = Path("Pop1K7/midi_analyzed")

# print(f"--- scanning MIDI files in {DATA_DIR} ... ---")
# midi_paths: List[Path] = list(DATA_DIR.glob("**/*.mid")) + list(DATA_DIR.glob("**/*.midi"))
# print(f"find {len(midi_paths)} MIDI files.")


# # --- 4. (可選) 訓練 Tokenizer (例如 BPE) ---
# USE_BPE = True
# VOCAB_SIZE = 10000 # BPE 詞彙表大小 (可自行調整)

# if USE_BPE:
#     print(f"--- 正在訓練 BPE (Vocab size: {VOCAB_SIZE})... ---")
#     tokenizer.train(
#         vocab_size=VOCAB_SIZE,
#         files_paths=midi_paths,
#         # 你可以加入 'WordPiece' 或 'Unigram' 
#         model_type="BPE" 
#     )
#     print("BPE 訓練完成。")

# # --- 5. 執行 Tokenization ---
# print("--- Tokenized ---")
# all_tokenized_data = []

# for i, midi_path in enumerate(midi_paths):
#     if (i + 1) % 100 == 0:
#         print(f"  processing... ({i + 1} / {len(midi_paths)})")
        
#     try:
#         # MidiTok Tokenizer 可以直接讀取檔案路徑
#         # 這會載入 MIDI -> 轉換為 Tokens -> 轉換為 IDs (一氣呵成)
#         token_ids = tokenizer(midi_path) 
#         all_tokenized_data.append(token_ids)
        
#     except Exception as e:
#         # 某些 MIDI 檔案可能已損毀或格式不符
#         print(f" warning: not handling {midi_path}: {e}")

# print(f"finished tokenizing {len(all_tokenized_data)} files.")


# # --- 6. 儲存處理好的資料和 Tokenizer ---

# # 儲存 Tokenizer (包含詞彙表和 BPE 模型)
# # 這樣你在訓練和推論時才能載入完全相同的設定
# tokenizer_save_path = "tokenizer/cpword_tokenizer.json"
# tokenizer.save_params(tokenizer_save_path)
# print(f"Tokenizer is stored at {tokenizer_save_path}")

# # 儲存 Tokenized 序列
# # 使用 pickle (與 tutorial.ipynb 相同)
# corpus_save_path = "tokenizer/cpword_tokenized_corpus.pkl"
# with open(corpus_save_path, "wb") as f:
#     pickle.dump(all_tokenized_data, f)
# print(f"Tokenized corpus is stored at {corpus_save_path}")

# print("\n--- end ---")
# correct_cpword_tokenizer.py
# remi_tokenizer.py
import pickle
from pathlib import Path
from typing import List
from miditok import TokenizerConfig, REMI
import numpy as np

# ===== 1. 使用 REMI 而不是 CPWord =====
tokenizer_config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},
    nb_velocities=32,
    use_chords=True,
    use_rests=True,
    use_rest_programs=True,
    use_tempos=True,
    use_time_signatures=False,
    use_programs=False,
    rest_range=(2, 8),
    nb_tempos=32,
    tempo_range=(40, 250),
)

tokenizer = REMI(tokenizer_config)
print(f"--- Tokenizer: REMI ---")
print(f"Base vocabulary size: {len(tokenizer)}")

# ===== 2. 載入 MIDI 檔案 =====
DATA_DIR = Path("Pop1K7/midi_analyzed")
print(f"\nScanning MIDI files in {DATA_DIR}...")

midi_paths: List[Path] = list(DATA_DIR.glob("**/*.mid")) + list(DATA_DIR.glob("**/*.midi"))
print(f"Found {len(midi_paths)} MIDI files")

# ===== 3. 測試單個文件 =====
print(f"\n{'='*50}")
print("Testing single file...")
print(f"{'='*50}")

test_file = midi_paths[0]
print(f"Test file: {test_file.name}")

tok_seq = tokenizer(test_file)
print(f"Type: {type(tok_seq)}")

if hasattr(tok_seq, 'ids'):
    token_ids = list(tok_seq.ids)
    print(f"✓ Token IDs length: {len(token_ids)}")
    print(f"  Type of IDs: {type(token_ids)}")
    print(f"  First 50 IDs: {token_ids[:50]}")
elif isinstance(tok_seq, list) and hasattr(tok_seq[0], 'ids'):
    token_ids = list(tok_seq[0].ids)
    print(f"✓ Token IDs length: {len(token_ids)}")
    print(f"  First 50 IDs: {token_ids[:50]}")

# ===== 4. 訓練 BPE =====
USE_BPE = False
VOCAB_SIZE = 3000

if USE_BPE:
    print(f"\n{'='*50}")
    print(f"Training BPE (vocab size: {VOCAB_SIZE})...")
    print(f"{'='*50}")
    tokenizer.train(
        vocab_size=VOCAB_SIZE,
        files_paths=midi_paths,
    )
    print(f"✓ BPE completed. Final vocab size: {len(tokenizer)}")

# ===== 5. 批量 Tokenization =====
print(f"\n{'='*50}")
print("Starting batch tokenization...")
print(f"{'='*50}")

all_tokenized_data = []
skipped_count = 0
error_count = 0

for i, midi_path in enumerate(midi_paths):
    if (i + 1) % 100 == 0:
        print(f"Processing... ({i + 1}/{len(midi_paths)})")
    
    try:
        tok_seq = tokenizer(midi_path)
        
        # 提取 IDs
        if hasattr(tok_seq, 'ids'):
            token_ids = list(tok_seq.ids)
        elif isinstance(tok_seq, list) and hasattr(tok_seq[0], 'ids'):
            token_ids = list(tok_seq[0].ids)
        else:
            skipped_count += 1
            continue
        
        # 過濾太短的序列
        if len(token_ids) < 100:
            skipped_count += 1
            continue
        
        # 確保是 int list
        token_ids = [int(t) for t in token_ids]
        all_tokenized_data.append(token_ids)
        
    except Exception as e:
        error_count += 1
        if error_count <= 5:
            print(f"  Error: {midi_path.name}: {e}")

# ===== 6. 統計 =====
print(f"\n{'='*50}")
print("Tokenization Summary:")
print(f"{'='*50}")
print(f"Total files: {len(midi_paths)}")
print(f"Successfully tokenized: {len(all_tokenized_data)}")
print(f"Skipped: {skipped_count}")
print(f"Errors: {error_count}")

if len(all_tokenized_data) == 0:
    print("\n✗ ERROR: No sequences were tokenized!")
    exit(1)

lengths = [len(seq) for seq in all_tokenized_data]
total_tokens = sum(lengths)

print(f"\nSequence Statistics:")
print(f"  Total tokens: {total_tokens:,}")
print(f"  Average length: {np.mean(lengths):.1f}")
print(f"  Median length: {np.median(lengths):.1f}")
print(f"  Min length: {min(lengths)}")
print(f"  Max length: {max(lengths)}")

print(f"\nSample sequences:")
for i in range(min(3, len(all_tokenized_data))):
    print(f"  Sequence {i+1}: length={len(all_tokenized_data[i])}")
    print(f"    First 50: {all_tokenized_data[i][:50]}")

# ===== 7. 儲存 =====
Path("tokenizer").mkdir(exist_ok=True)

tokenizer.save_params("tokenizer/remi_tokenizer.json")
print(f"\n✓ Tokenizer saved to: tokenizer/remi_tokenizer.json")

with open("tokenizer/remi_tokenized_corpus.pkl", "wb") as f:
    pickle.dump(all_tokenized_data, f)
print(f"✓ Corpus saved to: tokenizer/remi_tokenized_corpus.pkl")

print(f"\n{'='*50}")
print("✓ Done!")
print(f"{'='*50}")