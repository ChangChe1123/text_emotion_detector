import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import random
import re

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 设备配置 - 使用CPU（国内GPU版本下载较慢）
device = torch.device('cpu')
print(f"Using device: {device}")


# 简单的英文分词函数（避免使用torchtext）
def simple_tokenizer(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return tokens


# 1. 准备示例数据
texts = [
    "I love this movie, it's great!",
    "This film is terrible and boring.",
    "What an amazing performance by the actors.",
    "The plot was weak and disappointing.",
    "I enjoyed every moment of this film.",
    "Worst movie I've ever seen in my life.",
    "Brilliant direction and captivating story.",
    "The acting was poor and the script was bad.",
    "A masterpiece of modern cinema.",
    "I don't recommend this movie to anyone."
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative


# 2. 构建词汇表
def build_vocab(texts, special_tokens=['<pad>', '<unk>']):
    vocab = {}
    # 添加特殊token
    for idx, token in enumerate(special_tokens):
        vocab[token] = idx

    # 统计所有单词
    word_count = {}
    for text in texts:
        tokens = simple_tokenizer(text)
        for token in tokens:
            word_count[token] = word_count.get(token, 0) + 1

    # 按频率排序并添加到词汇表
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    for idx, (word, count) in enumerate(sorted_words):
        vocab[word] = idx + len(special_tokens)

    return vocab


vocab = build_vocab(texts)
vocab_size = len(vocab)


# 文本编码函数
def text_pipeline(text, vocab, max_length=20):
    tokens = simple_tokenizer(text)
    token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]

    # 填充或截断
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids = token_ids + [vocab['<pad>']] * (max_length - len(token_ids))

    return token_ids


# 3. 创建数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=20):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        text_ids = text_pipeline(text, self.vocab, self.max_length)

        return torch.tensor(text_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# 划分数据集
train_texts, test_texts = texts[:8], texts[8:]
train_labels, test_labels = labels[:8], labels[8:]

train_dataset = TextDataset(train_texts, train_labels, vocab)
test_dataset = TextDataset(test_texts, test_labels, vocab)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# 4. 定义简化版Transformer模型
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_length, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 位置编码（学习式）
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_length, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 词嵌入
        x = self.embedding(x)

        # 添加位置编码
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Transformer编码器
        x = self.transformer_encoder(x)

        # 使用第一个token的输出
        x = x[:, 0, :]

        # 分类层
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 模型参数
embed_dim = 64
num_heads = 4
hidden_dim = 128
num_layers = 2
num_classes = 2
max_length = 20

model = TransformerClassifier(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_length).to(
    device)

# 5. 训练配置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 6. 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


# 7. 评估和预测函数
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def predict_sentiment(text, model, vocab, max_length=20):
    model.eval()
    with torch.no_grad():
        token_ids = text_pipeline(text, vocab, max_length)
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        sentiment = "positive" if predicted_class == 1 else "negative"
        confidence = probabilities[0][predicted_class].item()

        return sentiment, confidence


# 8. 训练和评估
print("开始训练模型...")
train_model(model, train_loader, criterion, optimizer, num_epochs=20)

print("评估模型...")
evaluate_model(model, test_loader)

# 测试预测
test_text = "This movie is absolutely wonderful!"
sentiment, confidence = predict_sentiment(test_text, model, vocab)
print(f"文本: '{test_text}'")
print(f"情感: {sentiment}, 置信度: {confidence:.2f}")

# 保存模型
torch.save(model.state_dict(), 'transformer_classifier.pth')
print("模型已保存")