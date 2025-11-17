import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple

class MusicGenerationDataset(Dataset):
    """
    用於 Transformer-XL 的音樂生成 Dataset
    """
    
    def __init__(
        self,
        corpus_path: str,
        seq_len: int = 512,
        overlap: int = 64,
        min_seq_len: int = 32,
        pad_token_id: int = 0,
    ):
        self.seq_len = seq_len
        self.overlap = overlap
        self.pad_token_id = pad_token_id
        
        # 載入 tokenized corpus
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'rb') as f:
            raw_data = pickle.load(f)
        
        # 過濾太短的序列並展平
        print(f"Filtering sequences (min_len={min_seq_len})...")
        self.sequences = []
        total_tokens = 0
        
        for seq in raw_data:
            # 處理不同的資料格式並確保轉換為 list
            token_ids = self._extract_token_ids(seq)
            
            if token_ids is None:
                continue
            
            # 過濾太短的序列
            if len(token_ids) >= min_seq_len:
                self.sequences.append(token_ids)
                total_tokens += len(token_ids)
        
        print(f"Loaded {len(self.sequences)} sequences")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Average sequence length: {total_tokens / len(self.sequences):.1f}")
        
        # 建立 segments
        self.segments = self._create_segments()
    
    def _extract_token_ids(self, seq) -> List[int]:
        """
        從不同格式的資料中提取 token IDs 並確保返回 Python list
        """
        token_ids = None
        
        # 處理各種可能的格式
        if isinstance(seq, list):
            token_ids = seq
        elif isinstance(seq, np.ndarray):
            token_ids = seq.tolist()
        elif hasattr(seq, 'ids'):
            # MidiTok TokSequence 物件
            ids = seq.ids
            if isinstance(ids, list):
                token_ids = ids
            elif isinstance(ids, np.ndarray):
                token_ids = ids.tolist()
            else:
                token_ids = list(ids)
        elif hasattr(seq, 'tokens'):
            # 另一種可能的格式
            tokens = seq.tokens
            if isinstance(tokens, list):
                token_ids = tokens
            else:
                token_ids = list(tokens)
        else:
            # 嘗試直接轉換
            try:
                token_ids = list(seq)
            except:
                return None
        
        # 確保是 int 的 list
        if token_ids is not None:
            try:
                token_ids = [int(t) for t in token_ids]
            except:
                return None
        
        return token_ids
        
    def _create_segments(self) -> List[List[int]]:
        """
        將長序列切成多個 segments，確保返回 list of lists
        """
        segments = []
        stride = self.seq_len - self.overlap
        
        for seq in self.sequences:
            # 確保 seq 是 list
            if not isinstance(seq, list):
                seq = list(seq)
            
            seq_len = len(seq)
            
            if seq_len <= self.seq_len:
                # 序列較短，直接使用（確保是 list）
                segments.append(list(seq))
            else:
                # 序列較長，切成多個 segments
                for start_idx in range(0, seq_len - self.overlap, stride):
                    end_idx = start_idx + self.seq_len
                    if end_idx <= seq_len:
                        # 確保切片結果是 list
                        segments.append(list(seq[start_idx:end_idx]))
                    else:
                        # 最後一個 segment
                        segments.append(list(seq[start_idx:]))
        
        print(f"Created {len(segments)} segments from {len(self.sequences)} sequences")
        return segments
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
            input_ids: shape (seq_len-1,)
            target_ids: shape (seq_len-1,) - shifted by 1
        """
        segment = self.segments[idx]
        
        # 確保 segment 是 list
        if not isinstance(segment, list):
            segment = list(segment)
        
        # Padding 到固定長度
        if len(segment) < self.seq_len:
            segment = segment + [self.pad_token_id] * (self.seq_len - len(segment))
        elif len(segment) > self.seq_len:
            segment = segment[:self.seq_len]
        
        # 確保至少有 2 個 token（input 和 target）
        if len(segment) < 2:
            segment = [self.pad_token_id] * self.seq_len
        
        # 轉換為 tensor
        try:
            input_ids = torch.tensor(segment[:-1], dtype=torch.long)
            target_ids = torch.tensor(segment[1:], dtype=torch.long)
        except Exception as e:
            print(f"Error converting segment to tensor: {e}")
            print(f"Segment type: {type(segment)}")
            print(f"Segment length: {len(segment)}")
            print(f"First few items: {segment[:5] if len(segment) > 0 else 'empty'}")
            raise
        
        return input_ids, target_ids


def create_dataloaders(
    corpus_path: str,
    tokenizer_path: str,
    batch_size: int = 16,
    seq_len: int = 512,
    overlap: int = 64,
    num_workers: int = 4,
    # train_ratio: float = 0.95,
) -> Tuple[DataLoader, int]:
    """
    建立訓練和驗證的 DataLoader
    """
    from miditok import CPWord
    
    # 載入 tokenizer 來獲取 vocab size
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = CPWord(params=tokenizer_path)
    vocab_size = len(tokenizer)
    
    # 獲取 pad_token_id
    if hasattr(tokenizer, 'pad_token_id'):
        pad_token_id = tokenizer.pad_token_id
    elif hasattr(tokenizer, 'vocab'):
        # 嘗試從 vocab 中找 PAD token
        pad_token_id = tokenizer.vocab.get('PAD_None', 0)
    else:
        pad_token_id = 0
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Pad token ID: {pad_token_id}")
    
    # 建立 dataset
    full_dataset = MusicGenerationDataset(
        corpus_path=corpus_path,
        seq_len=seq_len,
        overlap=overlap,
        pad_token_id=pad_token_id,
    )
    
    
    # 建立 DataLoader
    # 重要：先用 num_workers=0 測試，確認沒問題後再增加
    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # 先設為 0 測試
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=0,  # 先設為 0 測試
    #     pin_memory=True,
    #     drop_last=False,
    # )
    
    print(f"\nDataLoader info:")
    print(f"  Training batches: {len(train_loader)}")
    # print(f"  Validation batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # return train_loader, val_loader, vocab_size
    return train_loader, vocab_size


# ===== 使用範例和測試 =====
if __name__ == "__main__":
    # 設定路徑
    CORPUS_PATH = "tokenizer/remi_tokenized_corpus.pkl"
    TOKENIZER_PATH = "tokenizer/remi_tokenizer.json"
    
    # 先測試載入資料
    print("="*50)
    print("Testing corpus loading...")
    print("="*50)
    
    with open(CORPUS_PATH, 'rb') as f:
        raw_data = pickle.load(f)
    
    print(f"Loaded {len(raw_data)} sequences")
    print(f"First sequence type: {type(raw_data[0])}")
    
    if hasattr(raw_data[0], 'ids'):
        print(f"First sequence has 'ids' attribute")
        print(f"  ids type: {type(raw_data[0].ids)}")
        print(f"  ids length: {len(raw_data[0].ids)}")
        print(f"  first 10 ids: {raw_data[0].ids[:10]}")
    elif isinstance(raw_data[0], list):
        print(f"First sequence is a list")
        print(f"  length: {len(raw_data[0])}")
        print(f"  first 10 items: {raw_data[0][:10]}")
    else:
        print(f"Unknown format: {raw_data[0]}")
    
    # 建立 dataloaders
    print("\n" + "="*50)
    print("Creating DataLoaders...")
    print("="*50)
    
    train_loader, val_loader, vocab_size = create_dataloaders(
        corpus_path=CORPUS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        batch_size=4,  # 小 batch size 用於測試
        seq_len=512,
        overlap=64,
        num_workers=0,  # 測試時用 0
    )
    
    # 測試 dataloader
    print("\n" + "="*50)
    print("Testing DataLoader...")
    print("="*50)
    
    try:
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Input shape: {input_ids.shape}")
            print(f"  Target shape: {target_ids.shape}")
            print(f"  Input dtype: {input_ids.dtype}")
            print(f"  Sample input tokens: {input_ids[0, :20].tolist()}")
            print(f"  Sample target tokens: {target_ids[0, :20].tolist()}")
            
            # 檢查資料有效性
            assert input_ids.shape[0] == target_ids.shape[0], "Batch size mismatch"
            assert input_ids.shape[1] == target_ids.shape[1], "Sequence length mismatch"
            assert input_ids.dtype == torch.long, "Wrong dtype"
            assert target_ids.dtype == torch.long, "Wrong dtype"
            
            if batch_idx >= 2:  # 只測試前 3 個 batch
                break
        
        print("\n" + "="*50)
        print("✓ DataLoader is working correctly!")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()