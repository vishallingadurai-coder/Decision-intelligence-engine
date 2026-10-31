# Online Retail Decision Intelligence Engine — Final Project Report

## 1. Abstract
This project builds an offline retail demand forecasting prototype using the uploaded Online Retail transaction dataset. It cleans transaction records, explores sales, aggregates product-day data, creates leakage-aware time-series features, trains a classifier for whether a product will sell and a log-transformed regression model for positive demand, evaluates on a chronological holdout period, and exports product-level next-day forecasts.

## 2. Business problem
Retail teams need to identify products that may sell soon and prioritise review. Transaction history can inform demand estimates, but does not show current inventory or supplier lead time. This project therefore creates demand signals for human review rather than verified reorder quantities.

## 3. Data and cleaning
The uploaded workbook contains 541,909 transaction rows and eight columns. The analysis excludes cancellations identified by invoice numbers beginning with `C`, rows missing essential fields, non-positive quantities, negative prices, and exact duplicates. The resulting positive-sales dataset has 526,052 rows, 3,941 distinct products, 20,726 invoices, and covers 2010-12-01 to 2011-12-09. Revenue is calculated as quantity multiplied by unit price. This cleaning approach is intended for demand modelling, not accounting reconciliation.

## 4. Exploratory data analysis
The pipeline exports daily revenue and top-product charts. Top products by revenue include StockCode DOT, 22423, 23843, 85123A, and 47566. Product 23843 has a total of 80,995 units in the cleaned data, illustrating the extreme values that can affect error metrics. See `figures/` and `outputs/top_products.csv`.

## 5. Target and feature engineering
The modelling unit is product × calendar day. Missing product-days within the observed period are filled with zero recorded sales. Each feature row uses information available through day *t* to predict quantity on day *t+1*. Features include product code, target-date calendar features, current and historical sales lags, and rolling means/maxima. A chronological split is used; the test period begins 2011-09-25.

## 6. Models
- **Baseline:** next-day quantity equals the most recently observed product-day quantity.
- **Classifier:** random forest estimates whether a product has any recorded sales on the next day. The probability threshold is selected from a validation slice of the training period using F1.
- **Regressor:** random forest trained on log1p of positive next-day quantities. Predictions are converted back with expm1.
- **Combined expected demand:** sale probability multiplied by conditional quantity prediction.

For practical runtime, the classifier uses a reproducible sample of up to 300,000 training rows; the regressor uses positive-demand training rows. The final chronological test period is not used to fit either model or select the threshold.

## 7. Results
| Metric | Result |
|---|---:|
| Baseline MAE | 8.156 |
| Combined model MAE | 5.865 |
| Baseline RMSE | 158.110 |
| Combined model RMSE | 154.011 |
| Classifier precision | 0.676 |
| Classifier recall | 0.673 |
| Classifier F1 | 0.674 |
| Zero-sales share in test target | 73.03% |

The combined model improves MAE by about 28.1% over the baseline on this split. RMSE remains high because it strongly penalises rare extreme values. A product-day with a target quantity of 80,995 is a dominant error and should be investigated rather than silently removed.

## 8. Decision support
The forecast output includes product code, description, forecast date, sale probability, conditional quantity, expected demand, and a readable review signal. The top-ranked products are relative prioritisation signals, not guarantees of future sales.

## 9. Limitations
- No stock-on-hand or supplier lead-time fields are available.
- Zero recorded sales does not prove zero customer demand or product availability.
- Extreme quantities create large RMSE.
- The model is an offline prototype, not deployed to a live business system.
- No automated purchase orders or verified stockout alerts are generated.

## 10. Future work
Add inventory snapshots, supplier lead times, purchase orders, margins and holding costs; compare seasonal-naive and boosting models; run walk-forward backtesting; produce forecast intervals; and build a dashboard only after the underlying model is validated.

## 11. Conclusion
The project connects retail transaction data to product-level demand estimates and review signals. On the held-out chronological period, the combined model improves MAE relative to a simple lag-1 baseline. The large RMSE shows that extreme demand events remain difficult and require further modelling and business context.
