# ======================================
# 🧠 Multimodal Price Prediction Pipeline (Ensemble) CLIP
# Text + Image + Brand + Extra Attributes → Regression (SMAPE)
# Ensemble: LightGBM + XGBoost + CatBoost
# ======================================

import torch
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import joblib

print("✅ Importing done")

# ======================================
# ⚙️ Configuration
# ======================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# ======================================
# 🔹 Load datasets
# ======================================
train_df = pd.read_csv("train_final_dataset_pitstop7486_finale_s3.csv")
test_df = pd.read_csv("test_final_dataset_pitstop7486_finale_s3.csv")
print(f"Loaded {len(train_df)} training samples")
print(f"Loaded {len(test_df)} test samples")







# ======================================
# 🔹 Load precomputed embeddings
# ======================================
train_text_embeddings = np.load("train_text_embeddings.npy", allow_pickle=True)
train_image_embeddings = np.load("train_image_embeddings.npy", allow_pickle=True)
train_brand_emb = np.load("train_brand_embeddings_s2.npy", allow_pickle=True)
train_extra_attr_emb = np.load("extra_attributes_embeddings_train.npy", allow_pickle=True)

test_text_embeddings = np.load("test_text_embeddings.npy", allow_pickle=True)
test_image_embeddings = np.load("test_image_embeddings.npy", allow_pickle=True)
test_brand_emb = np.load("test_brand_embeddings_s2.npy", allow_pickle=True)
test_extra_attr_emb = np.load("extra_attributes_embeddings.npy", allow_pickle=True)

print("✅ All embeddings loaded successfully")
print("Train text shape:", train_text_embeddings.shape)
print("Train image shape:", train_image_embeddings.shape)
print("Train brand shape:", train_brand_emb.shape)
print("Train extra attributes shape:", train_extra_attr_emb.shape)

# ======================================
# 🔹 Combine embeddings and numeric features
# ======================================
numeric_cols = ['total_value', 'unit_idx']  # Adjust as needed

X_train = np.hstack((
    train_text_embeddings,
    train_image_embeddings,
    train_brand_emb,
    train_extra_attr_emb,
    train_df[numeric_cols].values
))

y_train = train_df['price_log'].values

X_test = np.hstack((
    test_text_embeddings,
    test_image_embeddings,
    test_brand_emb,
    test_extra_attr_emb,
    test_df[numeric_cols].values
))

print("✅ Features combined successfully")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# ======================================
# 🔹 Remove rows with NaN/None from X_train
# ======================================
nan_mask = ~np.isnan(X_train).any(axis=1)
X_train = X_train[nan_mask]
y_train = y_train[nan_mask]

print(f"✅ Removed NaN rows — Remaining training samples: {X_train.shape[0]}")

# Convert embeddings + numeric features to float32 to save memory
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)

print("✅ Converted to float32 – memory optimized")


# ======================================
# ⚙️ TRAIN ENSEMBLE MODELS
# ======================================
print("\n🔹 Training LightGBM model...")
lgb_model = LGBMRegressor(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8
)
lgb_model.fit(X_train, y_train)

import torch, gc
torch.cuda.empty_cache()
gc.collect()

print("🔹 Training XGBoost model on CPU (to prevent CUDA OOM)...")

xgb_model = XGBRegressor(
    n_estimators=700,          # fewer trees to reduce RAM
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",        # CPU mode
    device="cpu",              # <--- force CPU
    n_jobs=8,                  # parallel CPU threads
    predictor="auto"
)
xgb_model.fit(X_train, y_train)


print("🔹 Training CatBoost model on CPU to prevent OOM...")

cat_model = CatBoostRegressor(
    iterations=700,      # reduce from 700
    learning_rate=0.03,
    depth=8,             # reduce from 8
    verbose=100,
    task_type="CPU"      # force CPU
)


cat_model.fit(X_train, y_train)

print("✅ All models trained successfully")

# ======================================
# 🔹 Predict on test set (ensemble)
# ======================================
y_pred_lgb = lgb_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_cat = cat_model.predict(X_test)

y_pred = (y_pred_lgb + y_pred_xgb +  y_pred_cat)/3.0
y_pred_final = np.exp(y_pred)  # convert back from log

# Round to 2 decimals
test_df['price'] = y_pred_final.round(2)
test_df[['sample_id', 'price']].to_csv("finale_output_pitstop_s3.csv", index=False)

print("✅ Ensemble predictions saved to finale_output_pitstop_s3.csv")

# ======================================
# 💾 Save models
# # ======================================
# joblib.dump(lgb_model, "lgb_model_s3.pkl")
# joblib.dump(xgb_model, "xgb_model_s3.pkl")
# cat_model.save_model("cat_model_s3.cbm")

print("💾 All models saved successfully!")
