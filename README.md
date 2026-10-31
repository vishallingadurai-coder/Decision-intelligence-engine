# Online Retail Decision Intelligence Engine

## Project status
The pipeline has been run against the uploaded `Online Retail.xlsx` dataset. The output metrics and forecasts are saved in `outputs/`. Re-run the notebook or script to reproduce them.

## Objective
Forecast next-calendar-day product demand and translate model outputs into explainable decision-support signals.

## Project structure
- `Decision_Intelligence_Engine.ipynb` — notebook entry point and result review
- `run_project.py` — complete reproducible training/evaluation pipeline
- `data/Online Retail.xlsx` — input dataset used for this run
- `outputs/evaluation_summary.json` — model metrics
- `outputs/test_predictions.csv` — chronological test predictions
- `outputs/next_day_product_forecast.csv` — next-day product forecasts
- `outputs/largest_prediction_errors.csv` — largest test errors for review
- `outputs/top_products.csv` — top products by revenue
- `outputs/threshold_validation.csv` — classifier threshold selection
- `models/` — saved classifier, regressor, and config
- `figures/` — generated EDA and evaluation charts
- `reports/Project_Report.md` — written project report
- `reports/FINAL_RUN_CHECKLIST.md` — final verification checklist

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open `Decision_Intelligence_Engine.ipynb` to review the saved results. To retrain models and regenerate outputs, uncomment the optional `%run run_project.py` line in the notebook or run `python run_project.py` from the project folder. Retraining can take several minutes and needs sufficient RAM. Trained model files are also supplied separately in `Decision_Intelligence_Engine_Trained_Models.zip`.

## Dataset cleaning
The pipeline standardises types, excludes cancellation invoices with an invoice number starting with `C`, keeps positive quantities, excludes negative prices, removes exact duplicate rows, and calculates revenue. This is a positive-sales view, not a full accounting ledger. CustomerID can remain missing and is not required by the demand model.

## Forecast target and features
The modelling unit is product × calendar day. Missing product-days inside the observed date range are filled with zero recorded sales. The target is the next calendar day's quantity. Features include product code, target weekday/month/day/weekend, current and historical sales lags, and rolling statistics. The test period is chronological and is not randomly mixed into training.

## Results from the uploaded dataset
- Raw transaction rows: 541,909
- Cleaned transaction rows: 526,052
- Products: 3,941
- Data period: 2010-12-01 to 2011-12-09
- Test split begins: 2011-09-25
- Test rows: 295,575
- Test rows with zero next-day sales: 73.03%
- Baseline MAE: 8.156
- Model expected-demand MAE: 5.865
- Baseline RMSE: 158.110
- Model expected-demand RMSE: 154.011
- Classifier precision / recall / F1: 0.676 / 0.673 / 0.674

MAE improved by about 28.1% compared with the lag-1 baseline on this test split. RMSE is still large because the data contains rare, very large quantities. One extreme product-day with quantity 80,995 is a major source of error.

## Important limitations
- A day with no transaction is treated as zero recorded sales; it does not prove the product was in stock or that demand was zero.
- No current inventory, supplier lead times, purchase orders, costs, or live feed are included.
- Decision labels are review signals, not verified stockout predictions or reorder quantities.
- Model training uses reproducible samples from the training period for practical runtime; the final test set remains chronological and untouched.
