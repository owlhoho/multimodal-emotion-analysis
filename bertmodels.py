import torch
import torch.nn as nn
from transformers import BertModel


class BERTEmotionClassifier(nn.Module):
    """BERT + 全连接层情感分类器"""

    def __init__(self, num_classes=7, dropout=0.1):
        super(BERTEmotionClassifier, self).__init__()

        # 预训练BERT模型
        self.bert = BertModel.from_pretrained('./bert-base-uncased')

        # 冻结BERT层（可选，加快训练）
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

        print(f"✅ 模型初始化完成")
        print(f"   BERT layers: {sum(1 for _ in self.bert.parameters())} 参数")
        print(f"   分类头: {sum(1 for p in [self.fc1, self.fc2] for _ in p.parameters())} 参数")

    def forward(self, input_ids, attention_mask):
        """
      Args:
          input_ids: (batch_size, seq_len)
          attention_mask: (batch_size, seq_len)

      Returns:
          logits: (batch_size, num_classes)
      """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 取[CLS]token的输出（句子表示）
        pooled_output = outputs.pooler_output  # (batch_size, 768)

        # 分类头
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)  # (batch_size, num_classes)

        return logits


class BERTWithLSTMEmotionClassifier(nn.Module):
    """BERT + LSTM + 全连接层情感分类器（更复杂的版本）"""

    def __init__(self, num_classes=7, dropout=0.1):
        super(BERTWithLSTMEmotionClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('./bert-base-uncased')

        # 添加LSTM层
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)  # 双向LSTM: 256*2=512
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, input_ids, attention_mask):
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 获取所有token的输出
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, 768)

        # 过LSTM
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)

        # 取最后一个timestep的输出
        last_hidden = lstm_output[:, -1, :]  # (batch_size, 512)

        # 分类头
        x = self.dropout(last_hidden)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits


if __name__ == '__main__':
    print("=" * 60)
    print("🧠 模型测试")
    print("=" * 60)

    # 创建模型
    model = BERTEmotionClassifier(num_classes=7)

    # 创建虚拟输入
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # 前向传播
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    print(f"\n✅ 模型输出shape: {logits.shape}")
    print(f"   应该是: torch.Size([{batch_size}, 7])")
    print(f"\n💡 一个样本的7个类别概率:")
    print(f"   {torch.softmax(logits[0], dim=0).detach().numpy()}")

    print("\n" + "=" * 60)