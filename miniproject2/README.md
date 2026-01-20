## 📁 Project Structure
``` bash
miniproject2
├── artifacts
│   ├── CatBoost_curves.png
│   ├── FM_curves.png
│   ├── LightGBM_curves.png
│   └── XGBoost_curves.png
├── inference.py
├── inference.sh
├── models
│   ├── catboost.cbm
│   ├── ensemble_meta.joblib
│   ├── feature_meta.joblib
│   ├── fm_model.pth
│   ├── lgbm_model.joblib
│   ├── te_maps.joblib
│   └── xgb_model.joblib
├── README.md
├── requirements.txt
├── src
│   └── utils.py
└── train.py
```

## 🧰 Environment Setup

Run

```bash
cd miniproject2
pip3 install -r requirements.txt
```

## 🚀 Training

Run the following commands:
``` bash
cd miniproject2
python3 train.py
```

This will train the model and save models and artifacts to:
``` bash
miniproject2/models/
miniproject2/artifacts
```

## 🔮 Prediction
To obtain the predictions **locally**, run:
```bash
cd miniproject2
python3 inference.py --test_path <path/to/test.csv> 
```
This will store predictions.csv to:
```bash
miniproject2/predictions.csv
```

To run predictions **on server**, run:
```bash
cd miniproject2
./inference.sh --test_path <path/to/test.csv/on/server> 
```
Likewise, this will store predictions.csv to: 
```bash
miniproject2/predictions.csv
```