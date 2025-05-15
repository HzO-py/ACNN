import pickle
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ====== Dataset Definition ======
class PPGDataset(Dataset):
    def __init__(self, samples, maxn=190.0, minn=47.0, pca=None):
        # Flatten each sample's features into a 1D array (84,)
        X_list = [s['features'].reshape(-1) for s in samples]
        X_np = np.stack(X_list, axis=0)                        # Shape: (N, 84)
        if pca is not None:
            # Apply PCA to reduce to (N, 10)
            X_np = pca.transform(X_np)                        

        # Targets: systolic and diastolic blood pressure (N, 2)
        y = np.array([[s['sbp'], s['dbp']] for s in samples], dtype=np.float32)
        self.maxn = maxn
        self.minn = minn
        self.rng = maxn - minn
        # Normalize targets to [0, 1]
        y_norm = (y - minn) / self.rng

        self.X = torch.tensor(X_np, dtype=torch.float32)
        self.y = torch.tensor(y_norm, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ====== MLP Model ======
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=8, output_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)

# ====== BHS Error Grading Function ======
def compute_bhs_metric(err, threshold):
    # err: array of shape (N, 2) containing the absolute errors
    # Returns percentage of errors within the threshold for each channel
    return np.mean(err <= threshold, axis=0) * 100

# ====== Training & Testing ======
def main(use_pca=False):
    # Load the pre-split 5-fold cross-validation data
    with open('../dataset/cross_validation_folds_bp_f.pkl', 'rb') as f:
        folds = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Collect results across folds
    me_all = []
    std_all = []
    mae_all = []
    bhs5_all = []
    bhs10_all = []
    bhs15_all = []
    val_losses = []

    maxn, minn = 190.0, 47.0
    rng = maxn - minn

    for fold_idx, fold in enumerate(folds, start=1):
        # PCA option
        pca = None
        if use_pca:
            train_X = np.stack([s['features'].reshape(-1) for s in fold['train']], axis=0)
            pca = PCA(n_components=10)
            pca.fit(train_X)

        # Create datasets and loaders
        train_ds = PPGDataset(fold['train'], maxn, minn, pca)
        val_ds   = PPGDataset(fold['val'],   maxn, minn, pca)
        test_ds  = PPGDataset(fold['test'],  maxn, minn, pca)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=32)
        test_loader  = DataLoader(test_ds,  batch_size=32)

        # Model, optimizer, and loss function
        model = MLP(input_size=train_ds.X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        best_state, best_val_loss = None, float('inf')
        # Training loop
        for epoch in range(1, 1001):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                loss = criterion(preds, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Validation
            model.eval()
            losses = []
            with torch.no_grad():
                for Xv, yv in val_loader:
                    Xv, yv = Xv.to(device), yv.to(device)
                    pv = model(Xv)
                    losses.append(criterion(pv, yv).item())
            avg_loss = np.mean(losses)
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_state = model.state_dict()
        val_losses.append(best_val_loss)

        # Testing
        model.load_state_dict(best_state)
        model.eval()
        preds_list, targets_list = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds_norm = model(Xb.to(device)).cpu().numpy()
                preds_list.append(preds_norm)
                targets_list.append(yb.numpy())
        preds = np.vstack(preds_list)
        trues = np.vstack(targets_list)

        # Denormalize errors
        errors = (preds - trues) * rng
        abs_err = np.abs(errors)

        # Compute metrics
        me = np.mean(errors, axis=0)
        sd = np.std(errors, axis=0)
        mae = np.mean(abs_err, axis=0)
        b5 = compute_bhs_metric(abs_err, 5)
        b10 = compute_bhs_metric(abs_err, 10)
        b15 = compute_bhs_metric(abs_err, 15)

        # Print results for this fold
        print(f"\n==== Fold {fold_idx} Test Results ====")
        print(f"ME (mmHg):   SBP={me[0]:+.2f}, DBP={me[1]:+.2f}")
        print(f"STD(mmHg):  SBP={sd[0]:.2f}, DBP={sd[1]:.2f}")
        print(f"MAE(mmHg):  SBP={mae[0]:.2f}, DBP={mae[1]:.2f}")
        print(f"BHS-5:      SBP={b5[0]:.1f}%, DBP={b5[1]:.1f}%")
        print(f"BHS-10:     SBP={b10[0]:.1f}%, DBP={b10[1]:.1f}%")
        print(f"BHS-15:     SBP={b15[0]:.1f}%, DBP={b15[1]:.1f}%")

        me_all.append(me)
        std_all.append(sd)
        mae_all.append(mae)
        bhs5_all.append(b5)
        bhs10_all.append(b10)
        bhs15_all.append(b15)

    # Cross-fold means and standard deviation summary
    me_all = np.array(me_all)
    std_all = np.array(std_all)
    mae_all = np.array(mae_all)
    bhs5_all = np.array(bhs5_all)
    bhs10_all = np.array(bhs10_all)
    bhs15_all = np.array(bhs15_all)

    print("\n===== Cross-Validation Summary =====")
    print(f"ME mean:    SBP={me_all[:,0].mean():+.2f}±{me_all[:,0].std():.2f}, DBP={me_all[:,1].mean():+.2f}±{me_all[:,1].std():.2f}")
    print(f"STD mean:   SBP={std_all[:,0].mean():.2f}±{std_all[:,0].std():.2f}, DBP={std_all[:,1].mean():.2f}±{std_all[:,1].std():.2f}")
    print(f"MAE mean:   SBP={mae_all[:,0].mean():.2f}±{mae_all[:,0].std():.2f}, DBP={mae_all[:,1].mean():.2f}±{mae_all[:,1].std():.2f}")
    print(f"BHS-5 mean: SBP={bhs5_all[:,0].mean():.1f}±{bhs5_all[:,0].std():.1f}%, DBP={bhs5_all[:,1].mean():.1f}±{bhs5_all[:,1].std():.1f}%")
    print(f"BHS-10 mean:SBP={bhs10_all[:,0].mean():.1f}±{bhs10_all[:,0].std():.1f}%, DBP={bhs10_all[:,1].mean():.1f}±{bhs10_all[:,1].std():.1f}%")
    print(f"BHS-15 mean:SBP={bhs15_all[:,0].mean():.1f}±{bhs15_all[:,0].std():.1f}%, DBP={bhs15_all[:,1].mean():.1f}±{bhs15_all[:,1].std():.1f}%")
    print(f"Avg Val Loss: {np.mean(val_losses):.4f}±{np.std(val_losses):.4f}")

if __name__ == '__main__':
    main(use_pca=False)
