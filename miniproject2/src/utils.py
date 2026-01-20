import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score, log_loss
import lightgbm as lgb


try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from catboost import CatBoostClassifier, Pool
    _HAS_CAT = True
except ImportError:
    _HAS_CAT = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False

def init_jieba():
    if not _HAS_JIEBA: return
    keywords = [
        "Python", "Java", "C++", "JavaScript", "React", "Vue", "SQL", "NoSQL",
        "深度學習", "機器學習", "人工智慧", "大數據", "資料科學", "專案管理",
        "行政人員", "業務人員", "行銷企劃", "社群小編", "儲備幹部", "服務生"
    ]
    for w in keywords: jieba.add_word(w)

if _HAS_JIEBA: init_jieba()

# ---------------- Text Helpers ----------------
def chinese_tokenize(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^\u4e00-\u9fff a-zA-Z0-9\+\#\.]", " ", text)
    if _HAS_JIEBA:
        return " ".join(jieba.lcut(text))
    else:
        text = re.sub(r"([\u4e00-\u9fff])", r" \1 ", text)
        return re.sub(r"\s+", " ", text).strip()

def safe_text(x):
    return str(x) if not pd.isnull(x) else ""

# ---------------- Feature Engineering ----------------
def build_feature_matrix(
    df: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer = None,
    svd_model: TruncatedSVD = None, 
    fit_vectorizer: bool = True,
    **kwargs
) -> Tuple[csr_matrix, Dict[str, Any]]:
    
    # 1. 文本特徵
    raw_job = (df['job_position0'].apply(safe_text) + " " + df['job_duty'].apply(safe_text))
    raw_talent = (df['talent_preference_position0'].apply(safe_text) + " " + df['talent_structured_skill'].apply(safe_text))
    
    job_text = raw_job.apply(chinese_tokenize)
    talent_text = raw_talent.apply(chinese_tokenize)
    
    # [Length]
    len_diff = (raw_job.apply(len) - raw_talent.apply(len)).values.reshape(-1, 1)
    
    # [Jaccard]
    def get_jaccard(text1, text2):
        s1 = set(text1.split())
        s2 = set(text2.split())
        if not s1 or not s2: return 0.0
        return len(s1 & s2) / len(s1 | s2)
    jaccard_scores = np.array([get_jaccard(j, t) for j, t in zip(job_text, talent_text)]).reshape(-1, 1)

    # [TF-IDF]
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=2500, 
            ngram_range=(1, 3), 
            stop_words=["的", "了", "和", "是", "未填寫", "未設定", "不拘", "無限制", "面試邀約", "doc", "請", "洽", "電洽"],
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        svd_model = TruncatedSVD(n_components=32, random_state=42)

    if fit_vectorizer:
        full_corpus = pd.concat([job_text, talent_text])
        tfidf_vectorizer.fit(full_corpus)
        X_sample = tfidf_vectorizer.transform(full_corpus.iloc[:50000])
        svd_model.fit(X_sample)
        
    X_job = tfidf_vectorizer.transform(job_text)
    X_talent = tfidf_vectorizer.transform(talent_text)
    X_job_svd = svd_model.transform(X_job)
    X_talent_svd = svd_model.transform(X_talent)
    
    svd_dist = np.linalg.norm(X_job_svd - X_talent_svd, axis=1).reshape(-1, 1)
    cosine_sim = X_job.multiply(X_talent).sum(axis=1)
    cosine_sim = np.array(cosine_sim).reshape(-1, 1)
    
    combined_text = (job_text + " " + talent_text)
    X_combined_tfidf = tfidf_vectorizer.transform(combined_text)

    # 2. 數值邏輯 (Smart Parsing)
    df_dense = pd.DataFrame(index=df.index)
    
    # (A) 薪資
    def parse_job_salary(s):
        s = str(s)
        if '面議' in s: return 0
        nums = re.findall(r'(\d{1,3}(?:,\d{3})*)', s)
        if not nums: return 0
        vals = [int(n.replace(',', '')) for n in nums]
        return max(vals)

    def parse_talent_worth(text):
        text = str(text)
        if pd.isnull(text) or text == '': return 0
        salaries = []
        monthly = re.findall(r'月薪\D*?(\d{1,3}(?:,\d{3})*)', text)
        for m in monthly:
            val = int(m.replace(',', ''))
            if val > 10000: salaries.append(val)
        hourly = re.findall(r'時薪\D*?(\d{1,3}(?:,\d{3})*)', text)
        for h in hourly:
            val = int(h.replace(',', ''))
            if val < 2000: salaries.append(val * 160)
        others = re.findall(r'未填寫薪資類別\D*?(\d{1,3}(?:,\d{3})*)', text)
        for o in others:
            val = int(o.replace(',', ''))
            if val > 10000: salaries.append(val)
        return max(salaries) if salaries else 0

    job_price = df['job_salary'].apply(parse_job_salary)
    talent_worth = df['talent_experience'].apply(parse_talent_worth)
    df_dense['val_salary_gap'] = np.where((job_price > 0) & (talent_worth > 0), job_price - talent_worth, 0)
    df_dense['num_salary'] = job_price
    df_dense['talent_worth'] = talent_worth

    # (B) 管理經驗
    def parse_management_exp(text):
        text = str(text)
        nums = re.findall(r'管理\D*?(\d+)[\s]*人', text)
        if not nums: return 0
        vals = [int(n) for n in nums]
        return max(vals)
    talent_max_team_size = df['talent_experience'].apply(parse_management_exp)
    
    def is_mgr(s): return 1 if '管理' in str(s) and '無需' not in str(s) else 0
    job_is_mgr = df['job_management'].apply(is_mgr)
    df_dense['feat_mgr_mismatch'] = ((job_is_mgr == 1) & (talent_max_team_size == 0)).astype(int)

    # (C) 公司規模
    df_dense['num_capital'] = df['org_capital'].apply(parse_job_salary)
    df_dense['num_staff'] = df['org_staff'].apply(parse_job_salary)

    # (D) 年資與穩定度
    def parse_tenure_stats(text):
        text = str(text)
        months = re.findall(r'\((\d+)\s*月經驗\)', text)
        if not months: return 0, 0, 0
        months = [int(m) for m in months]
        total_exp = sum(months) 
        job_count = len(months)
        avg_tenure = total_exp / job_count if job_count > 0 else 0
        return total_exp, job_count, avg_tenure
    tenure_stats = df['talent_experience'].apply(parse_tenure_stats)
    df_dense['talent_total_months'] = [x[0] for x in tenure_stats]
    df_dense['talent_job_count'] = [x[1] for x in tenure_stats]
    df_dense['talent_avg_tenure'] = [x[2] for x in tenure_stats]

    def parse_job_min_exp(text):
        text = str(text)
        if '不拘' in text or '未設定' in text: return 0
        nums = re.findall(r'(\d+)', text)
        return int(nums[0]) if nums else 0
    job_min_year = df['job_experience'].apply(parse_job_min_exp)
    talent_total_year = (df_dense['talent_total_months'] / 12).astype(int)
    df_dense['feat_exp_gap'] = talent_total_year - job_min_year

    # (E) 學歷
    edu_map = {'不拘': 0, '未設定': 0, '高中': 1, '專科': 2, '大學': 3, '碩士': 4, '博士': 5}
    def parse_edu(s):
        for k, v in edu_map.items():
            if k in str(s): return v
        return 0
    job_edu = df['job_grade'].apply(parse_edu)
    talent_edu = df['talent_education'].apply(parse_edu)
    df_dense['num_diff_edu'] = talent_edu - job_edu

    # (F) 人才基本屬性
    def parse_basic_info(text):
        text = str(text)
        age_match = re.search(r'年齡為\s*(\d+)', text)
        age = int(age_match.group(1)) if age_match else -1
        if '男性' in text: gender = 1
        elif '女性' in text: gender = 0
        else: gender = -1
        return age, gender
    basic_info = df['talent_basic'].apply(parse_basic_info)
    df_dense['talent_age'] = [x[0] for x in basic_info]
    df_dense['talent_gender'] = [x[1] for x in basic_info]

    # (G) 地點匹配
    job_city = df['job_workcity'].fillna("").astype(str)
    talent_city = df['talent_preference_city'].fillna("").astype(str)
    df_dense['feat_city_match'] = [1 if (t != "" and t in j) or (j in t) else 0 for j, t in zip(job_city, talent_city)]

    # (H) Count Encoding
    count_cols = ['org_name', 'job_position0', 'job_workcity', 'job_grade', 'org_trade']
    for col in count_cols:
        if col in df.columns:
            freq_map = df[col].value_counts().to_dict()
            df_dense[f'cnt_{col}'] = df[col].map(freq_map).fillna(0)

    # Target Encoding
    te_cols = [c for c in df.columns if c.startswith('te_')]
    if te_cols: df_dense = pd.concat([df_dense, df[te_cols]], axis=1)
        
    X_dense = csr_matrix(df_dense.fillna(0).values)
    
    # 3. 合併
    blocks = [
        X_combined_tfidf, 
        csr_matrix(cosine_sim), 
        csr_matrix(jaccard_scores), 
        csr_matrix(X_job_svd), 
        csr_matrix(X_talent_svd),
        csr_matrix(svd_dist),
        csr_matrix(len_diff),
        X_dense
    ]
    X = hstack(blocks, format='csr')
    
    meta = {
        'tfidf_vectorizer': tfidf_vectorizer,
        'svd_model': svd_model,
        'n_features': X.shape[1]
    }
    return X, meta

# ---------------- Ensemble Class ----------------
class EnsembleModel:
    def __init__(self, lgbm=None, xgb=None, cat=None, fm=None, weights=None):
        self.lgbm = lgbm
        self.xgb = xgb
        self.cat = cat
        self.fm = fm
        self.weights = weights if weights else [0.25, 0.25, 0.25, 0.25]
    
    def predict(self, X):
        final_pred = np.zeros(X.shape[0])
        w_lgbm, w_xgb, w_cat, w_fm = self.weights
        
        # 確保維度一致
        if len(self.weights) != 4:
             pass

        if self.lgbm and w_lgbm > 0:
            final_pred += w_lgbm * self.lgbm.predict(X)
        
        if self.xgb and w_xgb > 0:
            # XGBoost 需要 DMatrix
            dmat = xgb.DMatrix(X)
            final_pred += w_xgb * self.xgb.predict(dmat)
        
        if self.cat and w_cat > 0:
            # CatBoost predict_proba 返回 (N, 2)，取 index 1
            p_cat = self.cat.predict_proba(X)[:, 1]
            final_pred += w_cat * p_cat
        
        if self.fm and w_fm > 0 and _HAS_TORCH:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # FM 需要 Dense input 且 log1p 處理 (與 train 保持一致)
            # 假設 X 是 csr_matrix，需轉 dense
            # 如果記憶體不足，需改成分批預測
            try:
                X_dense = np.log1p(np.maximum(X.todense(), 0)).astype(np.float32)
                self.fm.eval()
                with torch.no_grad():
                    tensor_X = torch.from_numpy(X_dense).to(device)
                    p_fm = torch.sigmoid(self.fm(tensor_X)).cpu().numpy()
                    final_pred += w_fm * p_fm
            except Exception as e:
                print(f"[Ensemble] FM prediction failed: {e}")
                
        return final_pred
    
def compute_te_map(df, col_name, target_col='label', m=10):
    """計算並返回 Target Encoding 的 Mapping 字典"""
    global_mean = df[target_col].mean()
    stats = df.groupby(col_name)[target_col].agg(['count', 'mean'])
    smooth_val = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
    mapping = smooth_val.to_dict()
    return mapping, global_mean

def apply_te_map(df, col_name, mapping, global_mean):
    """將 Mapping 套用到 DataFrame"""
    feat_name = f'te_{col_name}'
    df[feat_name] = df[col_name].map(mapping).fillna(global_mean)
    return df

# ---------------- Target Encoding ----------------
def add_target_encoding(train_df, val_df, test_df, col_name, target_col='label', n_splits=5, m=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    feat_name = f'te_{col_name}'
    global_mean = train_df[target_col].mean()
    
    train_df[feat_name] = np.nan
    for train_idx, valid_idx in kf.split(train_df):
        X_tr, X_val = train_df.iloc[train_idx], train_df.iloc[valid_idx]
        stats = X_tr.groupby(col_name)[target_col].agg(['count', 'mean'])
        smooth_val = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
        train_df.loc[train_df.index[valid_idx], feat_name] = X_val[col_name].map(smooth_val)
    train_df[feat_name] = train_df[feat_name].fillna(global_mean)
    
    stats = train_df.groupby(col_name)[target_col].agg(['count', 'mean'])
    smooth_val = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
    
    val_df[feat_name] = val_df[col_name].map(smooth_val).fillna(global_mean)
    if test_df is not None:
        test_df[feat_name] = test_df[col_name].map(smooth_val).fillna(global_mean)
    return train_df, val_df, test_df

# ---------------- Models & Plotting ----------------

def plot_learning_curves(history: dict, model_name: str, save_dir='artifacts'):
    if not history:
        print(f"[{model_name}] No history to plot.")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    # 解析數據
    if model_name == 'LightGBM':
        # lgbm: {'training': {'auc': [], 'binary_logloss': []}, 'valid_1': {'auc': [], 'binary_logloss': []}}
        rounds = range(len(history['training']['binary_logloss']))
        loss_train = history['training']['binary_logloss']
        loss_val = history['valid_1']['binary_logloss']
        metric_train = history['training']['auc']
        metric_val = history['valid_1']['auc']
        metric_name = "AUC"
        
    elif model_name == 'XGBoost':
        # xgb: {'train': {'auc': [], 'logloss': []}, 'valid': {'auc': [], 'logloss': []}}
        rounds = range(len(history['train']['logloss']))
        loss_train = history['train']['logloss']
        loss_val = history['valid']['logloss']
        metric_train = history['train']['auc']
        metric_val = history['valid']['auc']
        metric_name = "AUC"
        
    elif model_name == 'CatBoost':
        # cat: {'learn': {'Logloss': []}, 'validation': {'Logloss': [], 'AUC': []}}
        # CatBoost 預設 learn 不一定會算 metric，視情況
        rounds = range(len(history['learn']['Logloss']))
        loss_train = history['learn']['Logloss']
        loss_val = history['validation']['Logloss']
        # 如果 learn 沒有 AUC，就只畫 validation 的
        metric_train = history['learn'].get('AUC', [0]*len(rounds)) 
        metric_val = history['validation']['AUC']
        metric_name = "AUC"

    elif model_name == 'FM':
        # custom: {'loss': [], 'val_loss': [], 'val_f1': []}
        rounds = range(len(history['loss']))
        loss_train = history['loss']
        loss_val = history['val_loss']
        metric_train = [0] * len(rounds) # FM 訓練迴圈沒算 train f1 節省時間
        metric_val = history['val_f1']
        metric_name = "F1-Score"
        
    else:
        return

    # 開始繪圖 (雙軸)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Epochs / Rounds')
    ax1.set_ylabel('BCE Loss', color='tab:red')
    ax1.plot(rounds, loss_train, label='Train Loss', color='tab:red', linestyle='--', alpha=0.5)
    ax1.plot(rounds, loss_val, label='Val Loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel(metric_name, color='tab:blue')
    ax2.plot(rounds, metric_train, label=f'Train {metric_name}', color='tab:blue', linestyle='--', alpha=0.5)
    ax2.plot(rounds, metric_val, label=f'Val {metric_name}', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title(f'{model_name} Training Dynamics')
    
    # 合併 Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_curves.png'))
    plt.close()
    print(f"[{model_name}] Curves saved to artifacts/")


def train_lgbm(X_train, y_train, X_val, y_val):
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    evals_result = {} # 用來存紀錄
    
    params = {
        'objective': 'binary', 'metric': ['auc', 'binary_logloss'], # 加入 logloss
        'learning_rate': 0.03, 'num_leaves': 63, 'max_depth': 12,
        'min_data_in_leaf': 100, 'min_gain_to_split': 0.05,
        'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 1,
        'lambda_l1': 5.0, 'lambda_l2': 5.0, 'verbose': -1,
        'seed': 42 
    }
    bst = lgb.train(
        params, dtrain, num_boost_round=3000, 
        valid_sets=[dtrain, dval], 
        callbacks=[
            lgb.early_stopping(150, first_metric_only=True), 
            lgb.log_evaluation(100),
            lgb.record_evaluation(evals_result)
        ]
    )
    return bst, evals_result

def train_cat(X_train, y_train, X_val, y_val):
    if not _HAS_CAT: return None, {}
    print("Training CatBoost...")
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)
    params = {
        'iterations': 2000, 'learning_rate': 0.03, 'depth': 6,
        'loss_function': 'Logloss', 'eval_metric': 'AUC',
        'early_stopping_rounds': 100, 'verbose': 100, 'allow_writing_files': False,
        'random_seed': 42
    }
    try:
        if _HAS_TORCH and torch.cuda.is_available(): params['task_type'] = 'GPU'
    except: pass
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool)
    return model, model.get_evals_result()

def train_xgb(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    evals_result = {}
    
    params = {
        'objective': 'binary:logistic', 'eval_metric': ['auc', 'logloss'], # 加入 logloss
        'learning_rate': 0.05, 'max_depth': 8,
        'subsample': 0.7, 'colsample_bytree': 0.7, 'verbosity': 0,
        'seed': 42
    }
    bst = xgb.train(
        params, dtrain, num_boost_round=2000, 
        evals=[(dtrain, 'train'), (dval, 'valid')],
        early_stopping_rounds=100, verbose_eval=False,
        evals_result=evals_result # 紀錄歷史
    )
    return bst, evals_result

if _HAS_TORCH:
    class FactorizationMachine(nn.Module):
        def __init__(self, input_dim, k=16):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)
            self.v = nn.Parameter(torch.randn(input_dim, k) * 0.01)
        def forward(self, x):
            linear_part = self.linear(x)
            inter_1 = torch.mm(x, self.v)
            inter_2 = torch.mm(x**2, self.v**2)
            interaction_part = 0.5 * torch.sum(inter_1**2 - inter_2, dim=1, keepdim=True)
            return (linear_part + interaction_part).squeeze(-1)

    def train_fm(X_train, y_train, X_val, y_val, batch_size=256, epochs=20, lr=5e-3):
        # 設定 torch seed
        torch.manual_seed(42)
        if torch.cuda.is_available(): torch.cuda.manual_seed(42)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[FM] Training on {device}...")
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        X_train = np.log1p(np.maximum(X_train.todense(), 0)).astype(np.float32)
        X_val = np.log1p(np.maximum(X_val.todense(), 0)).astype(np.float32)
        
        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32)))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.float32)))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        model = FactorizationMachine(input_dim=X_train.shape[1], k=16).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        # 紀錄歷史
        history = {'loss': [], 'val_loss': [], 'val_f1': []}

        best_auc = 0
        best_model_state = None
        patience = 4
        bad_epochs = 0
        
        for ep in range(epochs):
            model.train()
            epoch_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            history['loss'].append(epoch_loss / len(train_loader))

            model.eval()
            probs = []
            val_epoch_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_loss = criterion(out, yb)
                    val_epoch_loss += val_loss.item()
                    probs.append(torch.sigmoid(out).cpu().numpy())
            
            probs = np.concatenate(probs)
            auc = roc_auc_score(y_val, probs)
            f1 = f1_score(y_val, (probs >= 0.5).astype(int)) # 用 0.5 計算 F1
            
            history['val_loss'].append(val_epoch_loss / len(val_loader))
            history['val_f1'].append(f1)
            
            print(f"Epoch {ep+1}: Val AUC {auc:.4f}, Val F1 {f1:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_model_state = model.state_dict()
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience: break
        
        if best_model_state: model.load_state_dict(best_model_state)
        return model, history

# ---------------- Helpers ----------------
def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return {'precision': p, 'recall': r, 'f1': f, 'auc': roc_auc_score(y_true, y_prob)}

def find_best_threshold(y_true, y_prob):
    best_f1, best_thr = 0, 0.5
    for thr in np.linspace(0.1, 0.9, 100):
        f = f1_score(y_true, (y_prob >= thr).astype(int))
        if f > best_f1: best_f1, best_thr = f, thr
    return best_thr, best_f1

def save_model(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    print(f"Saved model to {path}")