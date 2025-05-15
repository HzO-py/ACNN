from multiCNN import *
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset


def load_folds(fold_path="../dataset/cross_validation_folds_bp_d.pkl"):
    with open(fold_path, "rb") as f:
        folds = pickle.load(f)
    return folds

##############################################
# 4. Training and evaluation functions (note input format conversion)
##############################################
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for (batch_inputs, labels) in dataloader:
        # batch_inputs: (B, 4, 2, 3, T)
        labels = labels.to(device)
        B, num_branches, num_cnn, C, T = batch_inputs.shape
        # The model requires input as a list of length num_branches,
        # each element is a list of length num_cnn,
        # each tensor has shape (B, 3, T)
        final_inputs = []
        for branch_idx in range(num_branches):
            branch_data = batch_inputs[:, branch_idx, :, :, :]  # (B, 2, 3, T)
            branch_list = list(torch.unbind(branch_data, dim=1))  # list of 2 tensors of shape (B, 3, T)
            final_inputs.append(branch_list)
        # Move inputs to device
        final_inputs = [[tensor.to(device) for tensor in branch] for branch in final_inputs]
        optimizer.zero_grad()
        outputs, _, _ = model(final_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (batch_inputs, labels) in dataloader:
            labels = labels.to(device)
            B, num_branches, num_cnn, C, T = batch_inputs.shape
            final_inputs = []
            for branch_idx in range(num_branches):
                branch_data = batch_inputs[:, branch_idx, :, :, :]
                branch_list = list(torch.unbind(branch_data, dim=1))
                final_inputs.append(branch_list)
            final_inputs = [[tensor.to(device) for tensor in branch] for branch in final_inputs]
            outputs, _, _ = model(final_inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
    return running_loss / len(dataloader.dataset)

##############################################
# 5. Compute BHS metric
##############################################
def compute_bhs_metric(errors, threshold):
    # errors: numpy array of shape (N, 2), threshold in mmHg
    return np.mean(errors <= threshold, axis=0) * 100  # return percentage for each metric

##############################################
# 6. Main training function: cross-validate across all folds,
#    include early stopping, and compute BHS metric
##############################################
def main_training_all_folds():
    folds = load_folds("../dataset/cross_validation_folds_bp_d.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    batch_size = 32
    patience = 100  # patience for early stopping
    fold_results = []
    
    # Train model for each fold
    for fold_idx, fold in enumerate(folds):
        print(f"\n===== Training Fold {fold_idx+1} =====")
        train_subjects = fold["train"]
        val_subjects = fold["val"]
        test_subjects = fold["test"]
        
        train_dataset = BP_Dataset_FourChannels_2CNN(train_subjects)
        val_dataset = BP_Dataset_FourChannels_2CNN(val_subjects)
        test_dataset = BP_Dataset_FourChannels_2CNN(test_subjects)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
        
        # Initialize model: num_branches=4, num_cnn=2, in_channels=3, out_dim=16
        model = MultiBranchPPGModel(num_branches=4, num_cnn=2, in_channels=3, out_dim=16).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"Fold {fold_idx+1} Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict()
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} for Fold {fold_idx+1}. Best Val Loss: {best_val_loss:.4f}")
                break
        
        fold_results.append(best_val_loss)
        # Save best model of current fold
        torch.save(best_state, f"../models/best_model_fold{fold_idx+1}.pth")
        
        # ======= Evaluate on test set =======
        model.load_state_dict(best_state)
        model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for (batch_inputs, labels) in test_loader:
                labels = labels.to(device)
                B, num_branches, num_cnn, C, T = batch_inputs.shape
                final_inputs = []
                for branch_idx in range(num_branches):
                    branch_data = batch_inputs[:, branch_idx, :, :, :]
                    branch_list = list(torch.unbind(branch_data, dim=1))
                    final_inputs.append(branch_list)
                final_inputs = [[tensor.to(device) for tensor in branch] for branch in final_inputs]
                outputs, _, _ = model(final_inputs)
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
        all_predictions = np.vstack(all_predictions)  # (N, 2)
        all_targets = np.vstack(all_targets)          # (N, 2)

        maxn, minn = 190, 47
        rng = maxn - minn

        abs_errors = np.abs((all_predictions - all_targets) * rng)  # (N, 2)
        mae_values = np.mean(abs_errors, axis=0)
        std_values = np.std(abs_errors, axis=0)
        bhs_5 = compute_bhs_metric(abs_errors, 5)
        bhs_10 = compute_bhs_metric(abs_errors, 10)
        bhs_15 = compute_bhs_metric(abs_errors, 15)
        
        print(f"\n==== Fold {fold_idx+1} Test Results ====")
        print(f"MAE: {mae_values} mmHg, STD: {std_values} mmHg")
        print(f"BHS-5: {bhs_5} %, BHS-10: {bhs_10} %, BHS-15: {bhs_15} %")
    
    avg_loss = sum(fold_results) / len(fold_results)
    print("\n===== Cross-Validation Training Completed =====")
    print(f"Average Best Validation Loss across all folds: {avg_loss:.4f}")

if __name__ == "__main__":
    main_training_all_folds()
