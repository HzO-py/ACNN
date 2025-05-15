import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

# ==========================================
# 1. Single-branch CNN feature extractor (customizable)
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, use_shortcut=False, use_dropout=False):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) if use_dropout else None
        self.use_shortcut = use_shortcut
        if use_shortcut and (in_channels != out_channels or stride != 1):
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.shortcut is not None:
            x = self.shortcut(x)
        if self.use_shortcut:
            out = out + x
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        return out

class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        """
        CNN feature extractor for each branch.
        Assumes input shape (B, in_channels, T) and outputs shape (B, 32).
        """
        super(CNNFeatureExtractor, self).__init__()
        self.block1 = ConvBlock(in_channels, 4, kernel_size=7, stride=2)
        self.pool1  = nn.MaxPool1d(kernel_size=7, stride=1)
        self.block2 = ConvBlock(4, 8, kernel_size=3, stride=1, use_dropout=True)
        self.block3 = ConvBlock(8, 8, kernel_size=3, stride=2, use_shortcut=True)
        self.block4 = ConvBlock(8, 16, kernel_size=3, stride=1, use_dropout=True)
        self.block5 = ConvBlock(16, 16, kernel_size=3, stride=2, use_shortcut=True)
        self.block6 = ConvBlock(16, 32, kernel_size=3, stride=1, use_dropout=True)
        self.block7 = ConvBlock(32, 32, kernel_size=3, stride=2, use_shortcut=True)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (B, in_channels, T)
        x = self.block1(x)      # (B, 4, T1)
        x = self.pool1(x)       # (B, 4, T2)  may reduce time dimension
        x = self.block2(x)      # (B, 8, T2)
        x = self.block3(x)      # (B, 8, T3)
        x = self.block4(x)      # (B, 16, T3)
        x = self.block5(x)      # (B, 16, T4)
        x = self.block6(x)      # (B, 32, T4)
        x = self.block7(x)      # (B, 32, T5)
        x = self.avgpool(x)     # (B, 32, 1)
        x = x.view(x.size(0), -1)  # (B, 32)
        return x

class TwoCNNCombine(nn.Module):
    def __init__(self, num_cnn=2, in_channels=3, out_dim=16):
        super(TwoCNNCombine, self).__init__()
        self.num_cnn = num_cnn
        # Create multiple CNN feature extractors
        self.extractors = nn.ModuleList([
            CNNFeatureExtractor(in_channels=in_channels) for _ in range(num_cnn)
        ])
        # Linear layer mapping concatenated features to out_dim
        self.fc = nn.Linear(32 * num_cnn, out_dim)

    def forward(self, inputs):
        """
        :param inputs: list/tuple of length num_cnn,
                       each tensor shape (B, in_channels, T)
        """
        if len(inputs) != self.num_cnn:
            raise ValueError(f"Expected {self.num_cnn} inputs, got {len(inputs)}")

        # Extract features from each CNN
        feats = [self.extractors[i](inputs[i]) for i in range(self.num_cnn)]  # each (B,32)

        # Concatenate -> (B, 32*num_cnn)
        x = torch.cat(feats, dim=1)

        # Fully connected + Sigmoid -> (B, out_dim)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x  # final representation Z

class MultiBranchAttentionFusion(nn.Module):
    """
    Attention fusion for N branch outputs:
      Input: (B, N, d), Output: fused feature (B, d) and attention weights (B, N)
    """
    def __init__(self, feature_dim):
        """:param feature_dim: dimension d of each branch feature"""
        super(MultiBranchAttentionFusion, self).__init__()
        # Project features to scalar scores
        self.attn_fc = nn.Linear(feature_dim, 1)

    def forward(self, x):
        """:param x: (B, N, d) -> returns fused (B, d), attn_weights (B, N)"""
        scores = self.attn_fc(x)           # (B, N, 1)
        attn_weights = F.softmax(scores, dim=1)  # along branch dim
        fused = (attn_weights * x).sum(dim=1)    # (B, d)
        attn_weights = attn_weights.squeeze(-1)  # (B, N)
        return fused, attn_weights

class MultiBranchPPGModel(nn.Module):
    def __init__(self, num_branches=4, num_cnn=2, in_channels=3, out_dim=16):
        super(MultiBranchPPGModel, self).__init__()
        self.num_branches = num_branches
        # Build multiple branches
        self.branch_extractors = nn.ModuleList([
            TwoCNNCombine(num_cnn=num_cnn, in_channels=in_channels, out_dim=out_dim)
            for _ in range(num_branches)
        ])
        # Attention fusion over branches
        self.attn_fusion = MultiBranchAttentionFusion(feature_dim=out_dim)
        # Final regression head (e.g., predict 2 values)
        self.regressor = nn.Linear(out_dim, 2)
    
    def forward(self, branch_inputs):
        """:param branch_inputs: list of length num_branches, each a list of num_cnn tensors"""
        if len(branch_inputs) != self.num_branches:
            raise ValueError(f"Input branch count {len(branch_inputs)} does not match expected {self.num_branches}!")
        branch_features = [
            self.branch_extractors[i](branch_inputs[i])
            for i in range(self.num_branches)
        ]  # list of (B, out_dim)
        features_stack = torch.stack(branch_features, dim=1)  # (B, num_branches, out_dim)
        fused_feature, attn_weights = self.attn_fusion(features_stack)
        regression_output = self.regressor(fused_feature)  # (B, 2)
        return regression_output, fused_feature, attn_weights

class BP_Dataset_FourChannels_2CNN(Dataset):
    def __init__(self, subjects):
        self.data = []
        for subject in subjects:
            label = np.array([subject["sbp"], subject["dbp"]], dtype=np.float32)
            for segment in subject["features"]:
                if len(segment) != 4:
                    continue
                channel_data = []
                for ch in range(4):
                    chan = segment[ch]
                    # Branch 1: raw + first + second derivatives
                    branch1 = np.stack([
                        chan["ppg_filtered"],
                        chan["ppg_first_derivative"],
                        chan["ppg_second_derivative"]
                    ], axis=0)
                    # Branch 2: envelopes
                    branch2 = np.stack([
                        chan["ppg_envelope"],
                        chan["ppg_first_envelope"],
                        chan["ppg_second_envelope"]
                    ], axis=0)
                    channel_data.append([branch1, branch2])
                # channel_data shape: (4, 2, 3, T)
                self.data.append((np.array(channel_data), label))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        branch_inputs, label = self.data[idx]
        return torch.tensor(branch_inputs, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    branch_inputs, labels = zip(*batch)
    branch_inputs = torch.stack(branch_inputs, dim=0)  # (B, 4, 2, 3, T)
    labels = torch.stack(labels, dim=0)                # (B, 2)
    maxn, minn = 190, 47
    rng = maxn - minn
    return branch_inputs, (labels - minn) / rng

# =========================
# 5. Test the end-to-end model
# =========================
if __name__ == "__main__":
    batch_size = 2
    in_channels = 3
    T = 2000
    num_branches = 4
    num_cnn = 2
    out_dim = 16

    # Prepare dummy inputs: list of 4 branches, each a list of 2 tensors
    branch_inputs = []
    for _ in range(num_branches):
        branch = [torch.randn(batch_size, in_channels, T) for _ in range(num_cnn)]
        branch_inputs.append(branch)
    
    model = MultiBranchPPGModel(num_branches=num_branches, num_cnn=num_cnn,
                                in_channels=in_channels, out_dim=out_dim)
    reg_output, fused_feature, attn_weights = model(branch_inputs)
    
    print("Regression output shape:", reg_output.shape)        # e.g., (B, 2)
    print("Fused feature shape:", fused_feature.shape)        # (B, out_dim)
    print("Attention weights shape:", attn_weights.shape)      # (B, num_branches)
