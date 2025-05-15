import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# Assuming you've already imported your previous model module
from multiCNN import MultiBranchPPGModel, BP_Dataset_FourChannels_2CNN, collate_fn

##############################################
# Function to load the model (handles DataParallel)
##############################################
def load_model(model_path, device, num_branches=4, num_cnn=2, in_channels=3, out_dim=16):
    model = MultiBranchPPGModel(
        num_branches=num_branches,
        num_cnn=num_cnn,
        in_channels=in_channels,
        out_dim=out_dim
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    # Remove the "module." prefix if DataParallel was used
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

##############################################
# Test a single model: run on the test set and compute metrics
##############################################
def test_model(model, test_loader, device, scale):
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for (batch_inputs, labels) in tqdm(test_loader, desc="Testing", leave=False):
            labels = labels.to(device)
            # batch_inputs: (B, 4, 2, 3, T)
            B, num_branches, num_cnn, C, T = batch_inputs.shape
            # Prepare input format: list of 4, each a list of 2 tensors of shape (B, 3, T)
            final_inputs = []
            for branch_idx in range(num_branches):
                branch_data = batch_inputs[:, branch_idx, :, :, :]  # (B, 2, 3, T)
                branch_list = list(torch.unbind(branch_data, dim=1))  # two tensors
                final_inputs.append(branch_list)
            # Move tensors to the device
            final_inputs = [[tensor.to(device) for tensor in branch] for branch in final_inputs]
            outputs, _, _ = model(final_inputs)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)  # shape: (N, 2)
    all_targets = np.vstack(all_targets)          # shape: (N, 2)
    
    # Compute errors (apply scale if unit conversion needed)
    errors = (all_predictions - all_targets) * scale  # signed errors
    abs_errors = np.abs(errors)                      # for MAE and BHS
    
    mae_values = np.mean(abs_errors, axis=0)
    me_values = np.mean(errors, axis=0)   # mean error (ME)
    std_values = np.std(errors, axis=0)
    
    def compute_bhs_metric(err, threshold):
        return np.mean(err <= threshold, axis=0) * 100
    
    bhs_5 = compute_bhs_metric(abs_errors, 5)
    bhs_10 = compute_bhs_metric(abs_errors, 10)
    bhs_15 = compute_bhs_metric(abs_errors, 15)
    
    return mae_values, me_values, std_values, bhs_5, bhs_10, bhs_15

##############################################
# Main: test all cross-validation folds
##############################################
def test_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_workers = 2
    maxn, minn = 190, 47
    rng = maxn - minn
    scale = rng  # scaling factor for unit conversion
    
    # Load cross-validation fold definitions (each fold has a "test" key)
    with open("../dataset/cross_validation_folds_bp_d.pkl", "rb") as f:
        folds = pickle.load(f)
    
    metrics_all = {
        'MAE': [], 'ME': [], 'STD': [],
        'BHS5': [], 'BHS10': [], 'BHS15': []
    }
    
    for fold_idx, fold in enumerate(folds):
        print(f"\n=== Testing Fold {fold_idx+1} ===")
        test_subjects = fold["test"]
        test_dataset = BP_Dataset_FourChannels_2CNN(test_subjects)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers
        )
        
        model_path = f"../models/best_model_fold{fold_idx+1}.pth"
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found. Skipping Fold {fold_idx+1}.")
            continue
        model = load_model(model_path, device)
        
        mae, me, std, bhs5, bhs10, bhs15 = test_model(model, test_loader, device, scale)
        print(f"ME (mmHg):   SBP={me[0]:+.2f}, DBP={me[1]:+.2f}")
        print(f"STD(mmHg):  SBP={std[0]:.2f}, DBP={std[1]:.2f}")
        print(f"MAE(mmHg):  SBP={mae[0]:.2f}, DBP={mae[1]:.2f}")
        print(f"BHS-5:      SBP={bhs5[0]:.1f}%, DBP={bhs5[1]:.1f}%")
        print(f"BHS-10:     SBP={bhs10[0]:.1f}%, DBP={bhs10[1]:.1f}%")
        print(f"BHS-15:     SBP={bhs15[0]:.1f}%, DBP={bhs15[1]:.1f}%")

        # Collect metrics
        metrics_all['MAE'].append(mae)
        metrics_all['ME'].append(me)
        metrics_all['STD'].append(std)
        metrics_all['BHS5'].append(bhs5)
        metrics_all['BHS10'].append(bhs10)
        metrics_all['BHS15'].append(bhs15)

    # Summary across folds
    print("\n===== Cross-Validation Summary =====")
    for key, vals in metrics_all.items():
        arr = np.array(vals)
        mean_sbp = arr[:,0].mean()
        std_sbp  = arr[:,0].std()
        mean_dbp = arr[:,1].mean()
        std_dbp  = arr[:,1].std()
        print(f"{key} mean: SBP={mean_sbp:.2f}±{std_sbp:.2f}, DBP={mean_dbp:.2f}±{std_dbp:.2f}")

if __name__ == "__main__":
    test_all_models()
