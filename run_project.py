from pathlib import Path
import json
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "Online Retail.xlsx"
OUT = BASE / "outputs"
MODELS = BASE / "models"
FIGURES = BASE / "figures"
for folder in [OUT, MODELS, FIGURES]:
    folder.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Place Online Retail.xlsx in the data folder.")

print("Loading dataset...")
raw = pd.read_excel(DATA_PATH, sheet_name="Online Retail")
df = raw.copy()
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
df["StockCode"] = df["StockCode"].astype("string").str.strip()
df["InvoiceNo"] = df["InvoiceNo"].astype("string").str.strip()

df = df.dropna(subset=["InvoiceDate", "InvoiceNo", "StockCode", "Quantity", "UnitPrice"]).copy()
df = df[~df["InvoiceNo"].str.upper().str.startswith("C", na=False)].copy()
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] >= 0)].copy()
df = df.drop_duplicates().copy()
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
df["SalesDate"] = df["InvoiceDate"].dt.floor("D")

print("Building product-day table...")
daily_agg = (df.groupby(["StockCode", "SalesDate"], as_index=False)
             .agg(QuantitySold=("Quantity", "sum"), Revenue=("Revenue", "sum")))
dates = pd.date_range(daily_agg["SalesDate"].min(), daily_agg["SalesDate"].max(), freq="D")
products = daily_agg["StockCode"].dropna().unique()
full_index = pd.MultiIndex.from_product([products, dates], names=["StockCode", "SalesDate"])
daily = (daily_agg.set_index(["StockCode", "SalesDate"])
         .reindex(full_index, fill_value=0).reset_index()
         .sort_values(["StockCode", "SalesDate"]).reset_index(drop=True))

# Feature row at date t predicts next-calendar-day quantity at t+1.
grouped = daily.groupby("StockCode", sort=False)["QuantitySold"]
daily["TargetNextDay"] = grouped.shift(-1)
daily["Lag1"] = daily["QuantitySold"]
daily["Lag7"] = grouped.shift(6)
daily["Lag14"] = grouped.shift(13)
daily["RollingMean7"] = grouped.transform(lambda s: s.rolling(7, min_periods=1).mean())
daily["RollingMean30"] = grouped.transform(lambda s: s.rolling(30, min_periods=1).mean())
daily["RollingMax7"] = grouped.transform(lambda s: s.rolling(7, min_periods=1).max())
target_date = daily["SalesDate"] + pd.Timedelta(days=1)
daily["DayOfWeek"] = target_date.dt.dayofweek
daily["Month"] = target_date.dt.month
daily["DayOfMonth"] = target_date.dt.day
daily["IsWeekend"] = (daily["DayOfWeek"] >= 5).astype(int)

feature_cols = [
    "StockCode", "DayOfWeek", "Month", "DayOfMonth", "IsWeekend",
    "Lag1", "Lag7", "Lag14", "RollingMean7", "RollingMean30", "RollingMax7"
]
model_data = daily.dropna(subset=["TargetNextDay"]).copy()
model_data["WillSellNextDay"] = (model_data["TargetNextDay"] > 0).astype(int)

unique_dates = np.sort(model_data["SalesDate"].unique())
split_date = pd.Timestamp(unique_dates[int(len(unique_dates) * 0.80)])
train = model_data[model_data["SalesDate"] < split_date].copy()
test = model_data[model_data["SalesDate"] >= split_date].copy()
assert train["SalesDate"].max() < test["SalesDate"].min()

train_dates = np.sort(train["SalesDate"].unique())
val_start = pd.Timestamp(train_dates[int(len(train_dates) * 0.80)])
subtrain = train[train["SalesDate"] < val_start]
valid = train[train["SalesDate"] >= val_start]

def sample_rows(frame, n):
    return frame if len(frame) <= n else frame.sample(n=n, random_state=RANDOM_STATE)

numeric_features = [c for c in feature_cols if c != "StockCode"]
def make_preprocessor():
    return ColumnTransformer([
        ("product", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), ["StockCode"]),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features)
    ])

print("Training classifier...")
classifier = Pipeline([
    ("prep", make_preprocessor()),
    ("model", RandomForestClassifier(
        n_estimators=80, max_depth=18, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
    ))
])
classifier.fit(sample_rows(subtrain, 200_000)[feature_cols],
               sample_rows(subtrain, 200_000)["WillSellNextDay"])
valid_sample = sample_rows(valid, 100_000)
valid_prob = classifier.predict_proba(valid_sample[feature_cols])[:, 1]
threshold_rows = []
for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (valid_prob >= threshold).astype(int)
    threshold_rows.append({
        "threshold": threshold,
        "precision": precision_score(valid_sample["WillSellNextDay"], pred, zero_division=0),
        "recall": recall_score(valid_sample["WillSellNextDay"], pred, zero_division=0),
        "f1": f1_score(valid_sample["WillSellNextDay"], pred, zero_division=0)
    })
threshold_table = pd.DataFrame(threshold_rows)
threshold_table.to_csv(OUT / "threshold_validation.csv", index=False)
best_threshold = float(threshold_table.loc[threshold_table["f1"].idxmax(), "threshold"])

# Refit classifier on a representative sample from the full training period.
train_sample = sample_rows(train, 300_000)
classifier.fit(train_sample[feature_cols], train_sample["WillSellNextDay"])

print("Training log-quantity regressor...")
positive_train = train[train["TargetNextDay"] > 0]
positive_train_sample = sample_rows(positive_train, 220_000)
regressor = Pipeline([
    ("prep", make_preprocessor()),
    ("model", RandomForestRegressor(
        n_estimators=100, max_depth=20, min_samples_leaf=4, max_features=0.9,
        random_state=RANDOM_STATE, n_jobs=-1
    ))
])
regressor.fit(positive_train_sample[feature_cols],
              np.log1p(positive_train_sample["TargetNextDay"].to_numpy()))

print("Evaluating on the untouched chronological test period...")
X_test = test[feature_cols]
y_test = test["TargetNextDay"].to_numpy()
sale_probability = classifier.predict_proba(X_test)[:, 1]
sale_prediction = (sale_probability >= best_threshold).astype(int)
conditional_quantity = np.maximum(0, np.expm1(regressor.predict(X_test)))
expected_demand = sale_probability * conditional_quantity
baseline_prediction = np.maximum(0, test["Lag1"].to_numpy())
positive_mask = y_test > 0

metrics = {
    "dataset_file": DATA_PATH.name,
    "raw_rows": int(len(raw)),
    "clean_rows": int(len(df)),
    "rows_removed": int(len(raw) - len(df)),
    "duplicate_rows_after_clean": int(df.duplicated().sum()),
    "products": int(df["StockCode"].nunique()),
    "date_start": str(df["SalesDate"].min().date()),
    "date_end": str(df["SalesDate"].max().date()),
    "revenue_positive_sales": float(df["Revenue"].sum()),
    "units_positive_sales": int(df["Quantity"].sum()),
    "invoices": int(df["InvoiceNo"].nunique()),
    "customers_with_id": int(df["CustomerID"].nunique()),
    "missing_customer_ids_after_clean": int(df["CustomerID"].isna().sum()),
    "split_date": str(split_date.date()),
    "train_rows": int(len(train)),
    "test_rows": int(len(test)),
    "test_zero_sales_share": float((y_test == 0).mean()),
    "baseline_mae": float(mean_absolute_error(y_test, baseline_prediction)),
    "baseline_rmse": float(mean_squared_error(y_test, baseline_prediction) ** 0.5),
    "expected_demand_mae": float(mean_absolute_error(y_test, expected_demand)),
    "expected_demand_rmse": float(mean_squared_error(y_test, expected_demand) ** 0.5),
    "positive_day_quantity_mae": float(mean_absolute_error(y_test[positive_mask], conditional_quantity[positive_mask])),
    "positive_day_quantity_rmse": float(mean_squared_error(y_test[positive_mask], conditional_quantity[positive_mask]) ** 0.5),
    "classifier_threshold": best_threshold,
    "classifier_precision": float(precision_score(test["WillSellNextDay"], sale_prediction, zero_division=0)),
    "classifier_recall": float(recall_score(test["WillSellNextDay"], sale_prediction, zero_division=0)),
    "classifier_f1": float(f1_score(test["WillSellNextDay"], sale_prediction, zero_division=0)),
    "regression_target_transform": "log1p; predictions converted back with expm1",
    "classifier_training_rows": int(len(train_sample)),
    "regressor_training_rows": int(len(positive_train_sample)),
    "model_note": "Training rows sampled for practical runtime; final test period remains chronological and untouched."
}
with open(OUT / "evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

results = test[["StockCode", "SalesDate", "TargetNextDay", "WillSellNextDay"]].copy()
results["ForecastDate"] = results["SalesDate"] + pd.Timedelta(days=1)
results["SaleProbability"] = sale_probability
results["PredictedWillSell"] = sale_prediction
results["ConditionalQuantity"] = conditional_quantity
results["ExpectedDemand"] = expected_demand
results["BaselineLag1"] = baseline_prediction
results["AbsoluteError"] = np.abs(results["TargetNextDay"] - results["ExpectedDemand"])
desc_map = df.groupby("StockCode")["Description"].agg(
    lambda s: s.dropna().mode().iloc[0] if not s.dropna().mode().empty else ""
)
results["Description"] = results["StockCode"].astype(str).map(desc_map)
results.to_csv(OUT / "test_predictions.csv", index=False)
results.nlargest(100, "AbsoluteError").to_csv(OUT / "largest_prediction_errors.csv", index=False)

print("Forecasting next calendar day...")
latest_date = daily["SalesDate"].max()
latest_rows = daily[daily["SalesDate"] == latest_date].copy()
latest_prob = classifier.predict_proba(latest_rows[feature_cols])[:, 1]
latest_qty = np.maximum(0, np.expm1(regressor.predict(latest_rows[feature_cols])))
forecast = pd.DataFrame({
    "StockCode": latest_rows["StockCode"].astype(str).to_numpy(),
    "ForecastDate": latest_date + pd.Timedelta(days=1),
    "SaleProbability": latest_prob,
    "ConditionalQuantity": latest_qty,
    "ExpectedDemand": latest_prob * latest_qty
})
forecast["Description"] = forecast["StockCode"].map(desc_map)
top_cutoff = float(forecast["ExpectedDemand"].quantile(0.95))
def decision(row):
    if row["SaleProbability"] >= best_threshold:
        return "High sale-likelihood signal — review availability"
    if row["ExpectedDemand"] >= top_cutoff and row["ExpectedDemand"] > 0:
        return "Top 5% expected-demand signal — review product"
    if row["ExpectedDemand"] < 1:
        return "Low expected demand — monitor normally"
    return "Uncertain — monitor"
forecast["DecisionSignal"] = forecast.apply(decision, axis=1)
forecast.sort_values("ExpectedDemand", ascending=False).to_csv(OUT / "next_day_product_forecast.csv", index=False)

print("Creating charts...")
daily_revenue = df.groupby("SalesDate")["Revenue"].sum().sort_index()
plt.figure(figsize=(11, 4))
daily_revenue.plot()
plt.title("Daily Revenue (Positive Sales Only)")
plt.xlabel("Date"); plt.ylabel("Revenue"); plt.tight_layout()
plt.savefig(FIGURES / "daily_revenue.png", dpi=150); plt.close()

top_products = (df.groupby("StockCode").agg(Revenue=("Revenue", "sum"), UnitsSold=("Quantity", "sum"))
                .sort_values("Revenue", ascending=False).head(10))
top_products.to_csv(OUT / "top_products.csv")
plt.figure(figsize=(9, 5))
top_products["Revenue"].sort_values().plot(kind="barh")
plt.title("Top 10 Products by Revenue"); plt.xlabel("Revenue"); plt.tight_layout()
plt.savefig(FIGURES / "top_products_revenue.png", dpi=150); plt.close()

plt.figure(figsize=(6, 5))
plt.scatter(np.minimum(y_test, 100), np.minimum(expected_demand, 100), alpha=0.15, s=8)
plt.xlabel("Actual next-day quantity (capped at 100)")
plt.ylabel("Predicted expected demand (capped at 100)")
plt.title("Actual vs Predicted Demand (visual cap only)")
plt.tight_layout(); plt.savefig(FIGURES / "actual_vs_predicted.png", dpi=150); plt.close()

joblib.dump(classifier, MODELS / "sale_classifier.joblib")
joblib.dump(regressor, MODELS / "log_quantity_regressor.joblib")
joblib.dump({
    "feature_cols": feature_cols,
    "sale_threshold": best_threshold,
    "split_date": str(split_date.date()),
    "last_observed_date": str(latest_date.date()),
    "forecast_date": str((latest_date + pd.Timedelta(days=1)).date())
}, MODELS / "project_config.joblib")

print(json.dumps(metrics, indent=2))
print(f"Done. Outputs saved in: {OUT}")
