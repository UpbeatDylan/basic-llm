"""
简单的分词器实现
"""
import re
from collections import Counter
from typing import List, Dict


class SimpleTokenizer:
    """简单的基于字符/词的分词器"""
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.bos_token = "<BOS>"
        
        # 特殊token的ID
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # 初始化特殊token
        self.word_to_idx[self.pad_token] = self.pad_token_id
        self.word_to_idx[self.unk_token] = self.unk_token_id
        self.word_to_idx[self.bos_token] = self.bos_token_id
        self.word_to_idx[self.eos_token] = self.eos_token_id
        
        self.idx_to_word[self.pad_token_id] = self.pad_token
        self.idx_to_word[self.unk_token_id] = self.unk_token
        self.idx_to_word[self.bos_token_id] = self.bos_token
        self.idx_to_word[self.eos_token_id] = self.eos_token
        
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """从文本中构建词汇表"""
        # 统计词频
        word_counts = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_counts.update(words)
        
        # 选择最常见的词
        most_common = word_counts.most_common(self.vocab_size - 4)  # 减去4个特殊token
        
        # 构建词汇表
        idx = 4  # 从4开始，因为0-3是特殊token
        for word, _ in most_common:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        self.is_fitted = True
    
    def _tokenize(self, text: str) -> List[str]:
        """将文本分词为词列表"""
        # 简单的分词：按空格和标点符号分割
        text = text.lower()
        # 保留字母、数字和基本标点
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return words
    
    def encode(self, text: str, add_special_tokens=True) -> List[int]:
        """将文本编码为token ID列表"""
        words = self._tokenize(text)
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for word in words:
            if word in self.word_to_idx:
                token_ids.append(self.word_to_idx[word])
            else:
                token_ids.append(self.unk_token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens=True) -> str:
        """将token ID列表解码为文本"""
        words = []
        for token_id in token_ids:
            if token_id in self.idx_to_word:
                word = self.idx_to_word[token_id]
                if skip_special_tokens and word in [self.pad_token, self.unk_token, 
                                                     self.eos_token, self.bos_token]:
                    if word == self.eos_token:
                        break
                    continue
                words.append(word)
        
        return ' '.join(words)
    
    def save(self, filepath: str):
        """保存分词器"""
        import json
        data = {
            'vocab_size': self.vocab_size,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': {int(k): v for k, v in self.idx_to_word.items()},
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """加载分词器"""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.word_to_idx = data['word_to_idx']
        self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}
        self.is_fitted = True


