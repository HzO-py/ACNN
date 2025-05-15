from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from model import *  # Ensure ACNN_BiLSTM model is imported
import torch.nn.functional as F
from tabulate import tabulate

# ✅ Set up device and multi-GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multi_gpu = torch.cuda.device_count() > 1  # Check if multiple GPUs are available
print(f"Using {torch.cuda.device_count()} GPUs" if multi_gpu else f"Using device: {device}")

# ✅ Define Dataset Class
class PPGDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # ✅ Load PPG spectrogram data from .npy file
        ppg = np.load(sample["ppg"], allow_pickle=True).astype(np.float32)  # Shape: (56, 127, 1000, 12)

        ppg = ppg/255.0
        # ✅ Convert to PyTorch tensor
        image = torch.tensor(ppg, dtype=torch.float32).permute(0,3,1,2)  # Shape: (56, 127, 1000, 12)

        image = F.interpolate(image, size=(127,256), mode='bilinear', align_corners=False)


        # ✅ Convert labels to tensor
        label = torch.tensor([sample["sbp"], sample["dbp"]], dtype=torch.float32)

        return image, label

def train_or_test(is_train):
    # ✅ Load dataset
    dataset = np.load("../dataset/bp.npy", allow_pickle=True)
    dataset = list(dataset)
    print(f"Total Subjects Loaded: {len(dataset)}")  # Should match dataset size

    # ✅ Train-Test Split (80% Train, 20% Test)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

    # ✅ Define PyTorch Dataset
    train_dataset = PPGDataset(train_data)
    test_dataset = PPGDataset(test_data)

    # ✅ 10-Fold Cross-Validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # ✅ Hyperparameters
    batch_size = 8 
    num_epochs = 50
    learning_rate = 0.001
    patience = 15  # ✅ Early stopping patience
    num_workers = 4  # ✅ Use multiple workers for faster data loading

    if is_train is True:
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            if fold==0:
                continue
            print(f"\n===== Fold {fold+1}/10 Training =====")

            # Create dataset subsets
            train_subset = Subset(train_dataset, train_idx)
            val_subset = Subset(train_dataset, val_idx)

            # Create DataLoaders
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

            # ✅ Initialize model
            model = ACNN_BiLSTM().to(device)
            
            # ✅ Multi-GPU Training
            if multi_gpu:
                model = nn.DataParallel(model)

            # Define loss function and optimizer
            criterion = nn.L1Loss()  # MAE for regression
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # ✅ Early Stopping Variables
            best_val_loss = float("inf")
            patience_counter = 0

            # Training loop
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0

                for batch in train_loader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)  # Outputs shape: (batch, 2)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)
                print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

                # Evaluate on Validation Set
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)

                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item()

                val_loss /= len(val_loader)
                print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

                # ✅ Early Stopping Check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0  # Reset patience counter
                    torch.save(model.module.state_dict() if multi_gpu else model.state_dict(), f"../models/best_acnn_bilstm_fold{fold+1}.pth")  # Save best model
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} for Fold {fold+1}. Best Validation Loss: {best_val_loss:.4f}")
                    break  # Stop training early

        print("\n===== Cross-Validation Training Completed! =====")

    # ✅ Evaluate on Test Set (20% of original data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ✅ Initialize Metrics Storage
    results = []

    # ✅ Evaluate Each Fold
    for fold in range(10):
        print(f"\nEvaluating Fold {fold+1}")
        print("-" * 30)

        # ✅ Load best model from fold
        model = ACNN_BiLSTM().to(device)
        if multi_gpu:
            model = torch.nn.DataParallel(model)

        state_dict = torch.load(f"../models/best_acnn_bilstm_fold{fold+1}.pth", map_location=device)

        # ✅ Remove "module." prefix if trained with DataParallel
        if multi_gpu:
            model.load_state_dict(state_dict)
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

        model.eval()

        # ✅ Initialize Metrics
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # ✅ Convert to NumPy Arrays
        all_predictions = np.vstack(all_predictions)  # Shape: (N, 2)
        all_targets = np.vstack(all_targets)  # Shape: (N, 2)

        # ✅ Compute Absolute Errors
        abs_errors = np.abs(all_predictions - all_targets)  # Shape: (N, 2)

        # ✅ Compute MAE and STD
        mae_values = np.mean(abs_errors, axis=0)  # Mean Absolute Error
        std_values = np.std(abs_errors, axis=0)   # Standard Deviation

        # ✅ Compute BHS-Metric (Percentage within X mmHg)
        def compute_bhs_metric(errors, threshold):
            return np.mean(errors <= threshold, axis=0) * 100

        bhs_5 = compute_bhs_metric(abs_errors, 5)
        bhs_10 = compute_bhs_metric(abs_errors, 10)
        bhs_15 = compute_bhs_metric(abs_errors, 15)

        # ✅ Store Results
        results.append({
            "fold": fold + 1,
            "mae": mae_values,
            "std": std_values,
            "bhs_5": bhs_5,
            "bhs_10": bhs_10,
            "bhs_15": bhs_15
        })

        # ✅ Print Results in Table Format
        print(tabulate([
            ["DBP", mae_values[0], std_values[0]],
            ["SBP", mae_values[1], std_values[1]],
        ], headers=["| Metric |", "MAE (mmHg)", "STD (mmHg)"], tablefmt="grid"))

        print("\n" + "-" * 36)
        print(tabulate([
            ["DBP", f"{bhs_5[0]:.1f} %", f"{bhs_10[0]:.1f} %", f"{bhs_15[0]:.1f} %"],
            ["SBP", f"{bhs_5[1]:.1f} %", f"{bhs_10[1]:.1f} %", f"{bhs_15[1]:.1f} %"],
        ], headers=["| Metric |", "<=5 mmHg", "<=10 mmHg", "<=15 mmHg"], tablefmt="grid"))

    print("\n===== Evaluation Completed for All Folds! =====")

if __name__ == "__main__":
    train_or_test(False)