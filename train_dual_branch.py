"""
train_dual_branch.py
====================
独立双支路训练脚本：TimesNet (时域) + STFT 2D-CNN (频域)
放在 Time-Series-Library 根目录下直接运行：
    python train_dual_branch.py

修复清单（相对于上一版本）：
  1. TimesNet 特征在分类层之前截取，不再用 5→64 的 hack
  2. STFT 提前预计算存成 .npy，训练时直接读，不在 DataLoader 里实时算
  3. 加 weighted CrossEntropyLoss，解决类别不平衡
  4. DummyArgs 补全 TimesNet 所有必要参数
  5. 评估改用 Macro F1，不只看 accuracy
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal as scipy_signal
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# ── TimesNet 路径（放在根目录下直接能找到）──────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.TimesNet import Model as TimesNetModel


# ==============================================================================
# Step 0: 提前预计算 STFT，存成 .npy（只需运行一次）
# ==============================================================================

# 每个通道的有效频率范围不同，按通道设置保留的 bin 数
CHANNEL_KEEP_BINS = [12, 8, 4, 4, 12, 12, 12, 8]
# 对应顺序:  BVP IBI EDA TEMP ACC_X ACC_Y ACC_Z HR
MAX_BINS = max(CHANNEL_KEEP_BINS)   # 12，所有通道 pad 到这个高度


def compute_stft_for_window(window_2d, fs=64, nperseg=64, noverlap=32):
    """
    单窗口 (seq_len, 8) → STFT tensor (8, MAX_BINS=12, n_frames)
    noverlap=32 即 50% 重叠，时间帧数约 59（1920 点 / 32 步长）
    """
    n_channels = window_2d.shape[1]
    spectrograms = []

    for c in range(n_channels):
        keep = CHANNEL_KEEP_BINS[c]
        sig = window_2d[:, c].astype(np.float64)

        _, _, Zxx = scipy_signal.stft(
            sig,
            fs=fs,
            window='hamming',
            nperseg=nperseg,
            noverlap=noverlap,
            boundary='zeros'
        )

        mag = np.abs(Zxx[:keep, :])                 # (keep, n_frames)
        log_mag = np.log(mag + 1e-8)

        # per-sample min-max 归一化 → [0, 1]
        mn, mx = log_mag.min(), log_mag.max()
        if mx > mn:
            log_mag = (log_mag - mn) / (mx - mn)

        # pad 到 MAX_BINS 行（不足的填 0）
        if keep < MAX_BINS:
            pad = np.zeros((MAX_BINS - keep, log_mag.shape[1]))
            log_mag = np.vstack([log_mag, pad])

        spectrograms.append(log_mag)                # (MAX_BINS, n_frames)

    return np.stack(spectrograms, axis=0).astype(np.float32)  # (8, 12, n_frames)


def precompute_stft(data_dir, splits=('train', 'val', 'test'), force=False):
    """
    把 X_train/val/test.npy 全部转成 X_train_stft/val_stft/test_stft.npy
    只需要运行一次；如果文件已存在则跳过（force=True 强制重算）。
    """
    for split in splits:
        src = os.path.join(data_dir, f'X_{split}.npy')
        dst = os.path.join(data_dir, f'X_{split}_stft.npy')

        if os.path.exists(dst) and not force:
            print(f'[STFT] {dst} 已存在，跳过（传 force=True 重新计算）')
            continue

        if not os.path.exists(src):
            print(f'[STFT] {src} 不存在，跳过')
            continue

        X = np.load(src)
        print(f'[STFT] 正在处理 {split}，共 {len(X)} 个窗口...')
        results = [compute_stft_for_window(X[i]) for i in range(len(X))]
        X_stft = np.stack(results, axis=0)          # (N, 8, 12, n_frames)
        np.save(dst, X_stft)
        print(f'[STFT] 保存完成：{dst}，shape={X_stft.shape}')


# ==============================================================================
# Step 1: Dataset（直接读预计算好的文件，速度快）
# ==============================================================================

class SleepDualDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.X_time = np.load(
            os.path.join(data_dir, f'X_{split}.npy')
        ).astype(np.float32)

        self.X_stft = np.load(
            os.path.join(data_dir, f'X_{split}_stft.npy')
        ).astype(np.float32)

        self.y = np.load(
            os.path.join(data_dir, f'y_{split}.npy')
        ).astype(np.int64)

        assert len(self.X_time) == len(self.X_stft) == len(self.y), \
            "时域、频域、标签数量不一致，请检查数据文件"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_time[idx]),   # (seq_len, 8)
            torch.from_numpy(self.X_stft[idx]),   # (8, 12, n_frames)
            self.y[idx]
        )


# ==============================================================================
# Step 2: 修改 TimesNet 让它暴露特征向量（在分类层之前截取）
# ==============================================================================

class TimesNetFeatureExtractor(nn.Module):
    """
    包裹 TimesNetModel，返回 projection 层之前的特征向量。
 
    根据源码，classification 流程是：
        enc_embedding → model[i] × e_layers → act → dropout
        → reshape (B, seq_len*d_model) → projection (B, num_class)
 
    我们在 reshape 之后、projection 之前截断，
    再加一个 Linear 降维到 feat_dim=128，避免参数爆炸。
    """
 
    def __init__(self, args, feat_dim=128):
        super().__init__()
        self.backbone   = TimesNetModel(args)
        self.feat_dim   = feat_dim
 
        # seq_len * d_model 可能高达 61440，先降维
        raw_dim = args.seq_len * args.d_model
        self.dim_reduce = nn.Sequential(
            nn.Linear(raw_dim, feat_dim),
            nn.ReLU()
        )
 
    def forward(self, x):
        """
        x : (B, seq_len, enc_in)
        返回: (B, feat_dim=128)
        """
        b = self.backbone
 
        # 1. embedding（classification 不需要 x_mark，传 None）
        enc_out = b.enc_embedding(x, None)          # (B, T, d_model)
 
        # 2. TimesBlock 堆叠（self.model 是 ModuleList）
        for i in range(b.layer):
            enc_out = b.layer_norm(b.model[i](enc_out))
 
        # 3. act + dropout（同原始 classification()）
        enc_out = b.act(enc_out)                    # gelu
        enc_out = b.dropout(enc_out)
 
        # 4. x_mark_enc 在原始代码里做 padding mask
        #    独立训练时没有 mask，用全1占位（等价于不 mask 任何位置）
        x_mark = torch.ones(x.shape[0], x.shape[1],
                            device=x.device, dtype=x.dtype)
        enc_out = enc_out * x_mark.unsqueeze(-1)    # (B, T, d_model)
 
        # 5. reshape → (B, seq_len * d_model)
        enc_out = enc_out.reshape(enc_out.shape[0], -1)
 
        # 6. 降维到 feat_dim（不经过原始 projection，保留特征语义）
        return self.dim_reduce(enc_out)             # (B, feat_dim)                            # (B, feature_dim)


# ==============================================================================
# Step 3: STFT 2D-CNN 支路
# ==============================================================================

class STFTBranch(nn.Module):
    def __init__(self, in_channels=8, out_features=128):
        super().__init__()
        # 输入: (B, 8, 12, ~59)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                  # → (B, 32, 6, ~29)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                  # → (B, 64, 3, ~14)

            nn.AdaptiveAvgPool2d((1, 1)),         # → (B, 64, 1, 1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                         # → (B, 64)
            nn.Linear(64, out_features),
            nn.ReLU()
        )

    def forward(self, x_stft):
        return self.fc(self.net(x_stft))          # (B, out_features)


# ==============================================================================
# Step 4: 双支路融合模型
# ==============================================================================

class DualBranchFusion(nn.Module):
    def __init__(self, timesnet_args, num_classes=5, stft_out=128):
        super().__init__()

        # 时域支路
        self.time_branch = TimesNetFeatureExtractor(timesnet_args, feat_dim=128)
        time_feat_dim = self.time_branch.feat_dim

        # 频域支路
        self.stft_branch = STFTBranch(in_channels=8, out_features=stft_out)

        # 融合分类头
        fusion_dim = time_feat_dim + stft_out
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_time, x_stft):
        F_t = self.time_branch(x_time)            # (B, time_feat_dim)
        F_f = self.stft_branch(x_stft)            # (B, stft_out)
        fused = torch.cat([F_t, F_f], dim=1)      # (B, time_feat_dim + stft_out)
        return self.classifier(fused)             # (B, num_classes)


# ==============================================================================
# Step 5: 训练 + 评估
# ==============================================================================

def get_weighted_criterion(y_train, num_classes=5, device='cuda'):
    """类别不平衡修复：少数类（N1/N3/R）自动获得更高权重"""
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=y_train
    )
    names = ['W', 'N1', 'N2', 'N3', 'R']
    print('\n[Loss] Weighted CrossEntropyLoss:')
    for n, w in zip(names, weights):
        print(f'  {n}: {w:.3f}')
    return nn.CrossEntropyLoss(
        weight=torch.FloatTensor(weights).to(device)
    )


def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x_t, x_s, y in loader:
            x_t, x_s = x_t.to(device), x_s.to(device)
            out = model(x_t, x_s)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(y.numpy())
    preds, trues = np.array(preds), np.array(trues)
    acc      = accuracy_score(trues, preds)
    macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
    return acc, macro_f1, preds, trues


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── 0. 预计算 STFT（首次运行会花几分钟，之后直接跳过）──
    precompute_stft(config['data_dir'])

    # ── 1. 数据 ──────────────────────────────────────────────
    train_ds = SleepDualDataset(config['data_dir'], 'train')
    val_ds   = SleepDualDataset(config['data_dir'], 'val')
    test_ds  = SleepDualDataset(config['data_dir'], 'test')

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'],
                              shuffle=False)

    print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}')
    print('Label dist (train):', Counter(train_ds.y.tolist()))

    # ── 2. 模型 ──────────────────────────────────────────────
    model = DualBranchFusion(
        timesnet_args=config['timesnet_args'],
        num_classes=5,
        stft_out=128
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {total_params:,}')

    # ── 3. Loss / Optimizer ──────────────────────────────────
    criterion = get_weighted_criterion(train_ds.y, device=device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                           weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    # ── 4. 训练循环 ──────────────────────────────────────────
    best_val_f1   = 0
    best_state    = None
    patience_cnt  = 0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        total_loss = 0

        for x_t, x_s, y in train_loader:
            x_t = x_t.to(device)
            x_s = x_s.to(device)
            y   = y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x_t, x_s), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step(val_f1)

        print(f'Epoch {epoch:3d} | '
              f'Loss {total_loss/len(train_loader):.4f} | '
              f'Val Acc {val_acc:.4f} | '
              f'Val Macro-F1 {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            print(f'  ✓ Best model saved (val F1={val_f1:.4f})')
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                print(f'Early stopping at epoch {epoch}')
                break

    # ── 5. 最终测试 ──────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    test_acc, test_f1, preds, trues = evaluate(model, test_loader, device)

    print('\n' + '='*50)
    print(f'Test Accuracy : {test_acc:.4f}')
    print(f'Test Macro-F1 : {test_f1:.4f}   ← 关键指标')
    print('='*50)
    print(classification_report(
        trues, preds,
        target_names=['W', 'N1', 'N2', 'N3', 'R'],
        zero_division=0
    ))

    # 保存结果
    os.makedirs('./dual_branch_results', exist_ok=True)
    np.save('./dual_branch_results/preds.npy', preds)
    np.save('./dual_branch_results/trues.npy', trues)
    torch.save(best_state, './dual_branch_results/best_model.pt')
    print('结果已保存到 ./dual_branch_results/')


# ==============================================================================
# 入口
# ==============================================================================

if __name__ == '__main__':

    class TimesNetArgs:
        task_name   = 'classification'
        seq_len     = 1920
        enc_in      = 8
        c_out       = 5
        num_class   = 5
        d_model     = 32       # 适合你的数据量，不要太大
        d_ff        = 64
        top_k       = 2        # BVP+ACC 两个主周期（心跳+呼吸）
        e_layers    = 2
        num_kernels = 6        # TimesNet inception kernel 数量
        dropout     = 0.3
        embed       = 'timeF'
        freq        = 'h'
        label_len = 0
        pred_len = 0

    config = {
        'data_dir'      : './dataset/sleep_data_ready/',
        'batch_size'    : 32,
        'epochs'        : 30,
        'lr'            : 1e-4,
        'patience'      : 7,
        'timesnet_args' : TimesNetArgs(),
    }

    train(config)
