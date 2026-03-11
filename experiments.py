"""
对比实验：测试不同方法的效果
"""
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import os
from dataset import get_dataloaders
from bertmodels import BERTEmotionClassifier, BERTWithLSTMEmotionClassifier
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_and_evaluate(model_name, model, train_loader, dev_loader, test_loader,
                       epochs=3, lr=2e-5, batch_size=16):
    """训练并评估模型"""

    print(f"\n{'=' * 70}")
    print(f"🧪 实验: {model_name}")
    print(f"{'=' * 70}")
    print(f"   学习率: {lr}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch Size: {batch_size}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_dev_acc = 0
    results = {
        'model_name': model_name,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'history': {
            'train_loss': [],
            'train_acc': [],
            'dev_acc': [],
            'test_acc': 0
        }
    }

    # 训练
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1} Train', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= len(train_loader)
        train_acc = train_correct / train_total

        # 验证
        model.eval()
        dev_correct = 0
        dev_total = 0

        with torch.no_grad():
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                dev_correct += (preds == labels).sum().item()
                dev_total += labels.size(0)

        dev_acc = dev_correct / dev_total

        results['history']['train_loss'].append(train_loss)
        results['history']['train_acc'].append(train_acc)
        results['history']['dev_acc'].append(dev_acc)

        print(f"   Epoch {epoch + 1}: Train Acc={train_acc:.4f}, Dev Acc={dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc

    # 测试
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = test_correct / test_total
    results['history']['test_acc'] = test_acc

    print(f"   ✅ 最终测试准确率: {test_acc:.4f}")

    return results


def main():
    print("=" * 70)
    print("🔬 MELD情感分析 - 对比实验")
    print("=" * 70)

    # 加载数据
    train_loader, dev_loader, test_loader, _ = get_dataloaders(batch_size=16)

    all_results = []

    # 实验1: 基础BERT + 小学习率
    model1 = BERTEmotionClassifier(num_classes=7)
    result1 = train_and_evaluate(
        model_name="BERT (lr=1e-5, epochs=3)",
        model=model1,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        epochs=3,
        lr=1e-5
    )
    all_results.append(result1)

    # 实验2: 基础BERT + 中等学习率
    model2 = BERTEmotionClassifier(num_classes=7)
    result2 = train_and_evaluate(
        model_name="BERT (lr=2e-5, epochs=3)",
        model=model2,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        epochs=3,
        lr=2e-5
    )
    all_results.append(result2)

    # 实验3: BERT + LSTM + 小学习率
    model3 = BERTWithLSTMEmotionClassifier(num_classes=7)
    result3 = train_and_evaluate(
        model_name="BERT+LSTM (lr=1e-5, epochs=3)",
        model=model3,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
        epochs=3,
        lr=1e-5
    )
    all_results.append(result3)

    # 结果对比
    print(f"\n{'=' * 70}")
    print("📊 实验结果对比")
    print(f"{'=' * 70}")

    print(f"\n{'模型':<30} {'学习率':<12} {'测试准确率':<15}")
    print("-" * 60)

    best_model = None
    best_acc = 0

    for result in all_results:
        name = result['model_name']
        lr = result['lr']
        test_acc = result['history']['test_acc']
        print(f"{name:<30} {lr:<12} {test_acc:<15.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = name

    print("-" * 60)
    print(f"\n🏆 最佳模型: {best_model} (Acc: {best_acc:.4f})")

    # 保存结果
    os.makedirs('results', exist_ok=True)
    with open('results/experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n✅ 结果已保存到 results/experiment_results.json")


if __name__ == '__main__':
    main()