import argparse
import os
import pandas as pd
import numpy as np
import joblib
import torch
import xgboost as xgb
from catboost import CatBoostClassifier

from src.utils import (
    build_feature_matrix, 
    apply_te_map, 
    EnsembleModel, 
    FactorizationMachine 
)

def main(args):
    print("--- Inference Start ---")
    
    # 1. 載入資料
    print(f"Loading test data from {args.test_path}...")
    try:
        # 自動偵測分隔符
        with open(args.test_path, encoding='utf-8') as f:
            line = f.read(1024)
            sep = ',' if ',' in line else '\t'
        test_df = pd.read_csv(args.test_path, sep=sep)
    except Exception as e:
        print(f"Error reading csv: {e}")
        return

    # 2. 載入 Metadata (Feature Engineering & Target Encoding)
    print("Loading metadata...")
    feature_meta = joblib.load('models/feature_meta.joblib')
    te_maps = joblib.load('models/te_maps.joblib')
    ensemble_meta = joblib.load('models/ensemble_meta.joblib')
    
    # 3. Apply Target Encoding
    print("Applying Target Encoding...")
    for col, data in te_maps.items():
        if col in test_df.columns:
            test_df = apply_te_map(test_df, col, data['map'], data['mean'])
        else:
            print(f"Warning: Column {col} missing in test set. Filling with global mean.")
            test_df[f'te_{col}'] = data['mean']

    # 4. Build Feature Matrix
    print("Building feature matrix...")
    X_test, _ = build_feature_matrix(
        test_df,
        tfidf_vectorizer=feature_meta['tfidf_vectorizer'],
        svd_model=feature_meta['svd_model'],
        fit_vectorizer=False # 不重新訓練 TF-IDF
    )
    print(f"Test Feature Shape: {X_test.shape}")

    # 5. Load Models
    print("Loading models...")
    
    # LightGBM
    try:
        bst_lgbm = joblib.load('models/lgbm_model.joblib')
    except:
        bst_lgbm = None
        
    # XGBoost
    try:
        bst_xgb = joblib.load('models/xgb_model.joblib')
    except:
        bst_xgb = None
        
    # CatBoost
    try:
        bst_cat = CatBoostClassifier()
        bst_cat.load_model('models/catboost.cbm')
    except:
        bst_cat = None
        
    # FM (Torch)
    bst_fm = None
    if os.path.exists('models/fm_model.pth'):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # 重新初始化模型結構
            input_dim = feature_meta['n_features']
            bst_fm = FactorizationMachine(input_dim=input_dim, k=16).to(device)
            bst_fm.load_state_dict(torch.load('models/fm_model.pth', map_location=device))
        except Exception as e:
            print(f"Failed to load FM model: {e}")

    # 6. Predict
    print("Predicting...")
    model = EnsembleModel(
        lgbm=bst_lgbm, 
        xgb=bst_xgb, 
        cat=bst_cat, 
        fm=bst_fm, 
        weights=ensemble_meta['weights']
    )
    
    test_probs = model.predict(X_test)
    threshold = ensemble_meta['threshold']
    
    # 產生結果
    predictions = (test_probs >= threshold).astype(int)
    
    output_df = pd.DataFrame({
        'ID': range(len(test_df)), 
        'Label': predictions
    })
    
    # 7. 儲存
    output_path = args.output_path
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', required=True, help='Path to test.csv')
    parser.add_argument('--output_path', default='predictions.csv', help='Path to save predictions')
    args = parser.parse_args()
    main(args)