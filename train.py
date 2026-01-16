"""
训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
from tqdm import tqdm
from model import MiniLLM
from tokenizer import SimpleTokenizer


class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # 截断或填充到固定长度
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        # 输入是除了最后一个token的所有token
        # 标签是除了第一个token的所有token（shift by 1）
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, labels


def load_data(data_path):
    """加载训练数据"""
    texts = []
    
    if os.path.isfile(data_path):
        # 如果是单个文件
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, list):
                    texts = data
                elif isinstance(data, dict):
                    texts = list(data.values())
            else:
                # 按行读取文本文件
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
    elif os.path.isdir(data_path):
        # 如果是目录，读取所有文本文件
        for filename in os.listdir(data_path):
            if filename.endswith(('.txt', '.json')):
                filepath = os.path.join(data_path, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    if filename.endswith('.json'):
                        data = json.load(f)
                        if isinstance(data, list):
                            texts.extend(data)
                    else:
                        texts.extend([line.strip() for line in f if line.strip()])
    
    return texts


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for input_ids, labels in pbar:
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # 计算损失（只计算非padding位置的损失）
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    # 配置参数
    config = {
        'data_path': 'data/train.txt',  # 训练数据路径
        'vocab_size': 10000,
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_len': 512,
        'dropout': 0.1,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'save_dir': 'checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_length': 512,
    }
    
    print(f"使用设备: {config['device']}")
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    texts = load_data(config['data_path'])
    print(f"加载了 {len(texts)} 条文本")
    
    # 创建分词器
    print("构建词汇表...")
    tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'])
    tokenizer.fit(texts)
    
    # 保存分词器
    tokenizer_path = os.path.join(config['save_dir'], 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    print(f"分词器已保存到: {tokenizer_path}")
    
    # 创建数据集和数据加载器
    dataset = TextDataset(texts, tokenizer, max_length=config['max_length'])
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    # 创建模型
    print("创建模型...")
    model = MiniLLM(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        pad_idx=tokenizer.pad_token_id
    ).to(config['device'])
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # 训练循环
    print("开始训练...")
    best_loss = float('inf')
    
    for epoch in range(1, config['num_epochs'] + 1):
        train_loss = train_epoch(model, dataloader, optimizer, criterion, 
                                config['device'], epoch)
        
        print(f"Epoch {epoch}/{config['num_epochs']}, Train Loss: {train_loss:.4f}")
        
        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'config': config,
        }
        
        checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_path = os.path.join(config['save_dir'], 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            print(f"保存最佳模型到: {best_model_path}")
    
    print("训练完成！")


if __name__ == '__main__':
    main()


