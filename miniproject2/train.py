import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import torch
import random
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from src.utils import (
    build_feature_matrix, add_target_encoding, 
    compute_te_map,
    train_lgbm, train_cat, train_fm, train_xgb,
    find_best_threshold, evaluate_predictions, save_model,
    plot_learning_curves, EnsembleModel
)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    seed_everything(42)
    os.makedirs('models', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(args.train_csv, sep=',' if ',' in open(args.train_csv, encoding='utf-8').read(100) else '\t')
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    # 1. Target Encoding
    print("Applying Target Encoding (Training)...")
    te_cols = ['org_name', 'job_position0', 'job_workcity', 'org_trade']
    for col in te_cols:
        train_df, val_df, _ = add_target_encoding(train_df, val_df, None, col)

    # 2. [重要] 建立並保存 Inference 用的 TE Mapping
    print("Saving Target Encoding Maps for Inference...")
    te_maps = {}
    for col in te_cols:
        mapping, global_mean = compute_te_map(train_df, col, 'label')
        te_maps[col] = {'map': mapping, 'mean': global_mean}
    save_model(te_maps, 'models/te_maps.joblib')

    # 建立特徵
    print("Building features...")
    X_train, meta = build_feature_matrix(train_df)
    save_model(meta, 'models/feature_meta.joblib') 
    
    X_val, _ = build_feature_matrix(
        val_df, 
        tfidf_vectorizer=meta['tfidf_vectorizer'], 
        svd_model=meta['svd_model'],
        fit_vectorizer=False
    )
    
    # Random Over Sampling
    print("Applying Random Over-sampling (ROS)...")
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, train_df['label'])
    
    # --- Training & Plotting & Saving ---
    
    # 1. LGBM
    print("\nTraining LightGBM (ROS)...")
    bst_lgbm, hist_lgbm = train_lgbm(X_train_ros, y_train_ros, X_val, val_df['label'])
    save_model(bst_lgbm, 'models/lgbm_model.joblib')
    plot_learning_curves(hist_lgbm, 'LightGBM')
    
    # 2. XGB
    print("\nTraining XGBoost (ROS)...")
    bst_xgb, hist_xgb = train_xgb(X_train_ros, y_train_ros, X_val, val_df['label'])
    save_model(bst_xgb, 'models/xgb_model.joblib')
    plot_learning_curves(hist_xgb, 'XGBoost')
    
    # 3. CatBoost
    print("\nTraining CatBoost (ROS)...")
    bst_cat, hist_cat = train_cat(X_train_ros, y_train_ros, X_val, val_df['label'])
    # CatBoost 有自己的 save_model 方法，但也可用 joblib
    bst_cat.save_model('models/catboost.cbm') 
    plot_learning_curves(hist_cat, 'CatBoost')
    
    # 4. FM
    bst_fm = None
    try:
        print("\nTraining FM (ROS)...")
        bst_fm, hist_fm = train_fm(X_train_ros, y_train_ros, X_val, val_df['label'].values)
        torch.save(bst_fm.state_dict(), 'models/fm_model.pth')
        plot_learning_curves(hist_fm, 'FM')
    except Exception as e:
        print(f"Skipping FM: {e}")

    # --- Ensemble ---
    te_maps = {}
    for col in ['org_name', 'job_position0', 'job_workcity', 'org_trade']:
        pass

    print("\nBuilding Ensemble...")
    weights = [0.4, 0.2, 0.2, 0.2] 
    model = EnsembleModel(bst_lgbm, bst_xgb, bst_cat, bst_fm, weights=weights)

    val_probs = model.predict(X_val)
    best_thr, best_f1 = find_best_threshold(val_df['label'], val_probs)
    print(f"\n[Result] Best Ensemble F1: {best_f1:.4f} at threshold {best_thr:.2f}")

    ensemble_meta = {
        'weights': weights,
        'threshold': best_thr,
        'f1_score': best_f1
    }
    save_model(ensemble_meta, 'models/ensemble_meta.joblib')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', default='train.csv')
    args = parser.parse_args()
    main(args)