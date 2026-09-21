import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter

# ==========================================
# 1. 极其轻量的 DataLoader (直接读算好的 .npy)
# ==========================================
class Dataset_STFT_Only(Dataset):
    def __init__(self, root_path, flag='train'):
        print(f"Loading {flag} STFT data from disk...")
        self.X_stft = np.load(os.path.join(root_path, f"X_{flag}_stft.npy")).astype(np.float32)
        self.y = np.load(os.path.join(root_path, f"y_{flag}.npy")).astype(np.int64)

    def __getitem__(self, index):
        return torch.tensor(self.X_stft[index]), torch.tensor(self.y[index])

    def __len__(self):
        return len(self.y)

# ==========================================
# 2. 纯 STFT 频域网络 (极其轻量的 2D-CNN)
# ==========================================
class STFT_Classifier(nn.Module):
    def __init__(self, in_channels=8, num_classes=5):
        super(STFT_Classifier, self).__init__()
        # 输入形状: [Batch, 8通道, 12频段, 31时间步]
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)) # 全局池化，压扁成 1x1
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # 展平: [Batch, 64]
        return self.classifier(x)

# ==========================================
# 3. 自动计算类别权重的 Loss
# ==========================================
def get_weighted_criterion(y_train, device):
    counter = Counter(y_train)
    total = sum(counter.values())
    num_classes = 5
    
    weights = np.zeros(num_classes)
    for cls in range(num_classes):
        if counter[cls] > 0:
            weights[cls] = total / (num_classes * counter[cls])
        else:
            weights[cls] = 1.0
            
    # 标准化权重
    weights = weights / np.sum(weights) * num_classes
    weights_tensor = torch.FloatTensor(weights).to(device)
    
    print("\n[Loss] Weighted CrossEntropyLoss:")
    class_names = ['W', 'N1', 'N2', 'N3', 'R']
    for i, w in enumerate(weights):
        print(f"  {class_names[i]}: {w:.3f}")
        
    return nn.CrossEntropyLoss(weight=weights_tensor)

# ==========================================
# 4. 极速训练循环
# ==========================================
def train():
    ROOT_PATH = './dataset/sleep_data_ready/'
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3 # CNN 可以用稍微大一点的学习率
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # 1. 准备数据
    train_dataset = Dataset_STFT_Only(ROOT_PATH, flag='train')
    val_dataset   = Dataset_STFT_Only(ROOT_PATH, flag='val')
    test_dataset  = Dataset_STFT_Only(ROOT_PATH, flag='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. 初始化
    model = STFT_Classifier().to(DEVICE)
    criterion = get_weighted_criterion(train_dataset.y, DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0.0

    # 3. 训练
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).view(-1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 验证
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).view(-1)
                outputs = model(x)
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_trues.extend(y.cpu().numpy())
                
        val_acc = accuracy_score(val_trues, val_preds)
        val_f1 = f1_score(val_trues, val_preds, average='macro')
        
        print(f"Epoch {epoch+1:3d} | Loss {train_loss/len(train_loader):.4f} | Val Acc {val_acc:.4f} | Val F1 {val_f1:.4f}", end="")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), './best_stft_only.pt')
            print("  ✓ Best saved", end="")
        print()

    # 4. 测试
    print("\n" + "="*50)
    print("Testing Best Model...")
    model.load_state_dict(torch.load('./best_stft_only.pt'))
    model.eval()
    
    test_preds, test_trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE).view(-1)
            outputs = model(x)
            test_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            test_trues.extend(y.cpu().numpy())

    print(f"Test Accuracy : {accuracy_score(test_trues, test_preds):.4f}")
    print(f"Test Macro-F1 : {f1_score(test_trues, test_preds, average='macro'):.4f}")
    print("="*50)
    
    target_names = ['W', 'N1', 'N2', 'N3', 'R']
    print(classification_report(test_trues, test_preds, target_names=target_names, zero_division=0))

if __name__ == '__main__':
    train()