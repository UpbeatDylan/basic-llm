"""
推理脚本
"""
import torch
import argparse
from model import MiniLLM
from tokenizer import SimpleTokenizer


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 加载分词器
    tokenizer = SimpleTokenizer()
    tokenizer_path = checkpoint_path.replace('.pt', '_tokenizer.json')
    # 尝试从检查点目录加载分词器
    import os
    tokenizer_dir = os.path.dirname(checkpoint_path)
    tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer.json')
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        print(f"警告: 未找到分词器文件 {tokenizer_path}")
        print("请确保分词器文件存在于检查点目录中")
        return None, None
    
    # 创建模型
    model = MiniLLM(
        vocab_size=len(tokenizer.word_to_idx),
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout'],
        pad_idx=tokenizer.pad_token_id
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, 
                  top_k=50, top_p=0.9, device='cpu'):
    """生成文本"""
    model.eval()
    
    # 编码输入
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = torch.tensor([tokens], device=device)
    
    generated = tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # 前向传播
            logits = model(generated)
            
            # 获取最后一个位置的logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k采样
            if top_k > 0:
                if next_token_logits.size(0) > top_k:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
            
            # Top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # 采样
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # 如果生成了结束符，停止
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # 解码
    generated_tokens = generated[0].cpu().tolist()
    # 移除输入部分，只保留生成的部分
    generated_tokens = generated_tokens[len(tokens[0]):]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text


def interactive_mode(model, tokenizer, device, max_length=100, temperature=1.0, 
                    top_k=50, top_p=0.9):
    """交互式生成模式"""
    print("\n进入交互模式，输入 'quit' 退出")
    print("=" * 50)
    
    while True:
        try:
            prompt = input("\n请输入提示文本: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式")
                break
            
            if not prompt:
                continue
            
            print("\n生成中...")
            generated = generate_text(
                model, tokenizer, prompt, max_length, 
                temperature, top_k, top_p, device
            )
            
            print(f"\n生成结果: {generated}")
            print("-" * 50)
        
        except KeyboardInterrupt:
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description='LLM推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--prompt', type=str, default=None,
                       help='输入提示文本（如果不提供则进入交互模式）')
    parser.add_argument('--max_length', type=int, default=100,
                       help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='温度参数（控制随机性）')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k采样参数')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p采样参数')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda/cpu)，默认自动选择')
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model, tokenizer = load_model(args.checkpoint, device)
    
    if model is None or tokenizer is None:
        print("模型加载失败")
        return
    
    print("模型加载成功！")
    
    # 生成文本
    if args.prompt:
        # 单次生成
        generated = generate_text(
            model, tokenizer, args.prompt, args.max_length,
            args.temperature, args.top_k, args.top_p, device
        )
        print(f"\n提示: {args.prompt}")
        print(f"生成: {generated}")
    else:
        # 交互模式
        interactive_mode(
            model, tokenizer, device, args.max_length,
            args.temperature, args.top_k, args.top_p
        )


if __name__ == '__main__':
    main()


