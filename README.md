# 小型LLM项目

这是一个完整的小型大语言模型（LLM）实现，包含所有基础组件模块、训练和推理代码。

## 项目结构

```
mini-llm/
├── model.py          # 模型架构（包含所有基础组件）
├── tokenizer.py      # 分词器实现
├── train.py          # 训练脚本
├── inference.py      # 推理脚本
├── config.json       # 配置文件
├── requirements.txt  # 依赖包
└── README.md         # 说明文档
```

## 模型架构

模型包含以下基础组件：

1. **位置编码 (PositionalEncoding)**: 为序列添加位置信息
2. **多头注意力 (MultiHeadAttention)**: 实现自注意力机制
3. **前馈网络 (FeedForward)**: 两层全连接网络
4. **层归一化 (LayerNorm)**: 归一化层
5. **解码器层 (DecoderLayer)**: 包含自注意力和前馈网络的完整层
6. **主模型 (MiniLLM)**: 完整的Transformer解码器架构

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备训练数据

创建 `data` 目录，并将训练文本文件放入其中。支持以下格式：
- 纯文本文件 (.txt)：每行一个样本
- JSON文件 (.json)：包含文本列表

示例：
```bash
mkdir data
# 将你的训练数据放入 data/train.txt
```

### 2. 训练模型

修改 `train.py` 中的 `data_path` 配置，然后运行：

```bash
python train.py
```

训练过程中会：
- 自动构建词汇表
- 保存分词器到 `checkpoints/tokenizer.json`
- 每个epoch保存检查点到 `checkpoints/checkpoint_epoch_N.pt`
- 保存最佳模型到 `checkpoints/best_model.pt`

### 3. 推理生成

#### 单次生成

```bash
python inference.py --checkpoint checkpoints/best_model.pt --prompt "你好，"
```

#### 交互模式

```bash
python inference.py --checkpoint checkpoints/best_model.pt
```

然后输入提示文本，模型会生成续写内容。

#### 自定义参数

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "今天天气" \
    --max_length 200 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9
```

参数说明：
- `--max_length`: 最大生成长度
- `--temperature`: 温度参数（值越大越随机，值越小越确定）
- `--top_k`: Top-k采样，只从概率最高的k个token中采样
- `--top_p`: Top-p采样（核采样），从累积概率达到p的token集合中采样

## 配置说明

可以在 `config.json` 中修改模型和训练参数：

- **model**: 模型架构参数
  - `vocab_size`: 词汇表大小
  - `d_model`: 模型维度
  - `n_heads`: 注意力头数
  - `n_layers`: 层数
  - `d_ff`: 前馈网络维度
  - `max_seq_len`: 最大序列长度
  - `dropout`: Dropout比率

- **training**: 训练参数
  - `batch_size`: 批次大小
  - `learning_rate`: 学习率
  - `num_epochs`: 训练轮数
  - `max_length`: 最大序列长度

## 模型特点

1. **完整的Transformer架构**: 实现了标准的解码器架构
2. **因果掩码**: 防止模型看到未来信息
3. **残差连接**: 每个子层都有残差连接
4. **层归一化**: 稳定训练过程
5. **多种采样策略**: 支持温度采样、Top-k和Top-p采样

## 注意事项

1. 训练数据需要足够大才能获得好的效果
2. 根据你的GPU内存调整 `batch_size` 和 `max_length`
3. 模型参数量取决于配置，可以通过调整 `d_model`、`n_layers` 等参数控制
4. 建议使用GPU训练，CPU训练会很慢

## 示例

训练一个简单的模型：

```python
# 1. 准备数据
# 创建 data/train.txt，每行一个句子

# 2. 训练
python train.py

# 3. 推理
python inference.py --checkpoint checkpoints/best_model.pt --prompt "人工智能"
```

## 扩展建议

- 添加学习率调度器
- 实现更高级的优化器（如AdamW with warmup）
- 添加验证集评估
- 实现更复杂的分词器（如BPE、SentencePiece）
- 添加模型评估指标（如困惑度）
- 实现梯度累积以支持更大的batch size


