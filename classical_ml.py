import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


def load_fold_data(fold, use_pca=False):
    X_train = np.stack([s['features'].reshape(-1) for s in fold['train']], axis=0)
    X_test  = np.stack([s['features'].reshape(-1) for s in fold['test']],  axis=0)
    y_train = np.array([[s['sbp'], s['dbp']] for s in fold['train']], dtype=np.float32)
    y_test  = np.array([[s['sbp'], s['dbp']] for s in fold['test']],  dtype=np.float32)
    if use_pca:
        pca = PCA(n_components=10)
        X_train = pca.fit_transform(X_train)
        X_test  = pca.transform(X_test)
    else:
        pca = None
    return X_train, y_train, X_test, y_test, pca


def evaluate_metrics(y_true, y_pred, rng):
    errors = (y_pred - y_true) * rng
    abs_err = np.abs(errors)
    me  = np.mean(errors, axis=0)
    sd  = np.std(errors, axis=0)
    mae = np.mean(abs_err, axis=0)
    bhs5  = np.mean(abs_err <= 5, axis=0) * 100
    bhs10 = np.mean(abs_err <= 10, axis=0) * 100
    bhs15 = np.mean(abs_err <= 15, axis=0) * 100
    return me, sd, mae, bhs5, bhs10, bhs15


def main(use_pca=False):
    with open('../dataset/cross_validation_folds_bp_f.pkl', 'rb') as f:
        folds = pickle.load(f)

    maxn, minn = 190.0, 47.0
    rng = maxn - minn

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'AdaBoost': AdaBoostRegressor(random_state=42)
    }

    for name, base in models.items():
        print(f"\n=== Model: {name} ===")
        me_list, sd_list, mae_list = [], [], []
        b5_list, b10_list, b15_list = [], [], []

        for idx, fold in enumerate(folds, start=1):
            X_train, y_train, X_test, y_test, pca = load_fold_data(fold, use_pca)
            y_train_n = (y_train - minn) / rng
            y_test_n  = (y_test  - minn) / rng

            model = MultiOutputRegressor(base)
            model.fit(X_train, y_train_n)
            pred_n = model.predict(X_test)

            me, sd, mae, b5, b10, b15 = evaluate_metrics(y_test_n, pred_n, rng)

            print(f"\n==== Fold {idx} Test Results ====")
            print(f"ME (mmHg):   SBP={me[0]:+.2f}, DBP={me[1]:+.2f}")
            print(f"STD(mmHg):  SBP={sd[0]:.2f}, DBP={sd[1]:.2f}")
            print(f"MAE(mmHg):  SBP={mae[0]:.2f}, DBP={mae[1]:.2f}")
            print(f"BHS-5:      SBP={b5[0]:.1f}%, DBP={b5[1]:.1f}%")
            print(f"BHS-10:     SBP={b10[0]:.1f}%, DBP={b10[1]:.1f}%")
            print(f"BHS-15:     SBP={b15[0]:.1f}%, DBP={b15[1]:.1f}%")

            me_list.append(me)
            sd_list.append(sd)
            mae_list.append(mae)
            b5_list.append(b5)
            b10_list.append(b10)
            b15_list.append(b15)

        me_arr  = np.array(me_list)
        sd_arr  = np.array(sd_list)
        mae_arr = np.array(mae_list)
        b5_arr  = np.array(b5_list)
        b10_arr = np.array(b10_list)
        b15_arr = np.array(b15_list)

        print("\n--- Cross-Validation Summary ---")
        print(f"ME:   SBP={me_arr[:,0].mean():+.2f}±{me_arr[:,0].std():.2f}, DBP={me_arr[:,1].mean():+.2f}±{me_arr[:,1].std():.2f}")
        print(f"STD:  SBP={sd_arr[:,0].mean():.2f}±{sd_arr[:,0].std():.2f}, DBP={sd_arr[:,1].mean():.2f}±{sd_arr[:,1].std():.2f}")
        print(f"MAE:  SBP={mae_arr[:,0].mean():.2f}±{mae_arr[:,0].std():.2f}, DBP={mae_arr[:,1].mean():.2f}±{mae_arr[:,1].std():.2f}")
        print(f"BHS-5:  SBP={b5_arr[:,0].mean():.1f}±{b5_arr[:,0].std():.1f}%, DBP={b5_arr[:,1].mean():.1f}±{b5_arr[:,1].std():.1f}%")
        print(f"BHS-10: SBP={b10_arr[:,0].mean():.1f}±{b10_arr[:,0].std():.1f}%, DBP={b10_arr[:,1].mean():.1f}±{b10_arr[:,1].std():.1f}%")
        print(f"BHS-15: SBP={b15_arr[:,0].mean():.1f}±{b15_arr[:,0].std():.1f}%, DBP={b15_arr[:,1].mean():.1f}±{b15_arr[:,1].std():.1f}%")

if __name__ == '__main__':
    main(use_pca=False)
