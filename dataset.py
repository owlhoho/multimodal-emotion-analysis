import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os


class MELDDataset(Dataset):
    """MELD情感分析数据集"""

    def __init__(self, csv_path, tokenizer, max_len=128):
        """
        Args:
            csv_path: CSV文件路径
            tokenizer: BERT分词器
            max_len: 最大序列长度
        """
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 情感标签映射
        self.emotion2id = {
            'neutral': 0,
            'joy': 1,
            'surprise': 2,
            'anger': 3,
            'sadness': 4,
            'disgust': 5,
            'fear': 6
        }

        # 情感ID映射（反向）
        self.id2emotion = {v: k for k, v in self.emotion2id.items()}

        print(f"✅ 数据集加载: {len(self.df)} 条样本")
        print(f"   情感类别: {list(self.emotion2id.keys())}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """获取单个样本"""
        row = self.df.iloc[idx]

        # 获取文本和标签
        text = str(row['Utterance'])
        emotion = row['Emotion']
        label = self.emotion2id[emotion]

        # 用BERT分词
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text,
            'emotion': emotion
        }


def get_dataloaders(batch_size=32, max_len=128):
    """创建数据加载器"""

    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

    # 创建数据集
    train_dataset = MELDDataset(
        'data/MELD/train_sent_emo.csv',
        tokenizer,
        max_len=max_len
    )

    dev_dataset = MELDDataset(
        'data/MELD/dev_sent_emo.csv',
        tokenizer,
        max_len=max_len
    )

    test_dataset = MELDDataset(
        'data/MELD/test_sent_emo.csv',
        tokenizer,
        max_len=max_len
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, dev_loader, test_loader, train_dataset


if __name__ == '__main__':
    # 测试数据加载
    train_loader, dev_loader, test_loader, dataset = get_dataloaders(batch_size=16)

    print("\n" + "=" * 60)
    print("🔍 数据加载测试")
    print("=" * 60)

    print(f"✅ 训练集: {len(train_loader)} batches")
    print(f"✅ 验证集: {len(dev_loader)} batches")
    print(f"✅ 测试集: {len(test_loader)} batches")

    # 看一个batch
    batch = next(iter(train_loader))
    print(f"\n📦 一个batch的内容:")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   attention_mask shape: {batch['attention_mask'].shape}")
    print(f"   label shape: {batch['label'].shape}")
    print(f"   样本文本: {batch['text'][0]}")
    print(f"   样本情感: {batch['emotion'][0]}")

    print("\n" + "=" * 60)