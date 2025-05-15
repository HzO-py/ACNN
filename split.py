import pickle
import random
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm


def split():
    # Assume 'dataset' already contains valid subject information
    # dataset = [ { "id": i, "sbp": ..., "dbp": ..., "ppg_segments": [...] }, ... ]
    dataset = np.load("../dataset/bp_f.npy", allow_pickle=True)  # Load the previously saved data
    dataset = list(dataset)  # Ensure it's a list

    # Use KFold to split subjects into 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = []

    # kf.split takes an iterable; here we use 'dataset' where each element is one subject
    for fold_index, (train_val_idx, test_idx) in enumerate(kf.split(dataset)):
        # Build test set from test_idx subjects
        test_set = [dataset[idx] for idx in test_idx]

        # The remainder is training+validation
        train_val_set = [dataset[idx] for idx in train_val_idx]

        # For reproducibility, set random_state when splitting train/val
        indices = list(range(len(train_val_set)))
        train_idx, val_idx = train_test_split(indices, test_size=0.25, random_state=42)
        train_set = [train_val_set[idx] for idx in train_idx]
        val_set = [train_val_set[idx] for idx in val_idx]

        print(f"Fold {fold_index+1}: Train={len(train_set)}, Validation={len(val_set)}, Test={len(test_set)}")

        folds.append({
            "train": train_set,
            "val": val_set,
            "test": test_set
        })

    # Save the 5-fold dataset splits
    with open("../dataset/cross_validation_folds_bp_f.pkl", "wb") as f:
        pickle.dump(folds, f)

    print("Cross-validation folds saved to '../dataset/cross_validation_folds_bp_f.pkl'.")


if __name__ == "__main__":
    split()
