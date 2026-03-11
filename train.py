import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
from dataset import get_dataloaders

from bertmodels import BERTEmotionClassifier
import json
from datetime import datetime

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  使用设备: {device}")

# 超参数
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_LEN = 128
WEIGHT_DECAY = 0.01

# 创建结果文件夹
os.makedirs('results', exist_ok=True)
os.makedirs('results/checkpoints', exist_ok=True)


def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        # 数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算准确率
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': correct / total})

    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """验证/测试"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def main():
    print("=" * 70)
    print("🚀 开始训练MELD情感分析模型")
    print("=" * 70)

    # 加载数据
    print("\n📥 加载数据...")
    train_loader, dev_loader, test_loader, dataset = get_dataloaders(
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN
    )

    # 创建模型
    print("\n🧠 创建模型...")
    model = BERTEmotionClassifier(num_classes=7)
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 训练记录
    history = {
        'train_loss': [],
        'train_acc': [],
        'dev_loss': [],
        'dev_acc': [],
        'best_dev_acc': 0
    }

    # 训练循环
    print(f"\n⏳ 开始训练 ({EPOCHS} epochs)...\n")

    for epoch in range(EPOCHS):
        print(f"\n{'=' * 70}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'=' * 70}")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)
        history['dev_loss'].append(dev_loss)
        history['dev_acc'].append(dev_acc)

        print(f"\n📊 Epoch {epoch + 1} 结果:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   Dev Loss: {dev_loss:.4f}, Dev Acc: {dev_acc:.4f}")

        # 保存最好的模型
        if dev_acc > history['best_dev_acc']:
            history['best_dev_acc'] = dev_acc
            checkpoint_path = f'results/checkpoints/best_model.pt'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"   ✅ 保存最好模型 (Acc: {dev_acc:.4f})")

    # 测试
    print(f"\n{'=' * 70}")
    print("🧪 在测试集上评估...")
    print(f"{'=' * 70}")

    # 加载最好的模型
    model.load_state_dict(torch.load('results/checkpoints/best_model.pt'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\n✅ 测试结果:")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Acc: {test_acc:.4f}")

    # 保存训练历史
    with open('results/history.json', 'w') as f:
        json.dump(history, f, indent=4)

    # 保存最终模型
    torch.save(model.state_dict(), 'results/final_model.pt')

    print(f"\n{'=' * 70}")
    print("✅ 训练完成！")
    print(f"{'=' * 70}")
    print(f"最佳验证精度: {history['best_dev_acc']:.4f}")
    print(f"测试精度: {test_acc:.4f}")
    print(f"\n📁 结果保存在 results/ 文件夹")


if __name__ == '__main__':
    main()