# DSS301 Model Component Notes

## Role scope for Data Scientist

Phat's model component turns the cleaned weather data and MobileNetV2 image embeddings into a decision-support output for agricultural UAV scheduling. The goal is not to build a computer-vision model from scratch. Instead, the project reuses pretrained image features and scikit-learn decision models, matching the lecturer's instruction that existing AI libraries/models are acceptable.

The demo is a hybrid DSS:

- Transparent safety rules act as guardrails for severe rain, dangerous weather codes, and excessive wind.
- The trained model learns the recommendation policy and supports comparison with other algorithms.
- If the ML suggestion conflicts with a hard safety rule, the DSS guardrail overrides it. This keeps the operator in control and avoids presenting the system as autonomous flight control.

## Modeling approach

Input features:

- Weather API variables: temperature, humidity, rain probability, precipitation, cloud cover, visibility, wind speed, wind gust, WMO weather code, hour, day of week, month.
- Image AI variables: `img_feature_0` to `img_feature_1279`, extracted by pretrained MobileNetV2.

Target label:

- `TAKE_OFF`: weather is acceptable for UAV operation.
- `DELAY_FLIGHT`: risk is moderate, usually rain probability or heat-related.
- `LOCK_SPRAY`: wind or wind gust is unsafe, so spraying should be blocked to reduce pesticide drift.
- `RETURN_TO_CHARGING`: rain or dangerous weather code makes operation unsafe.

Compared models:

- Majority baseline.
- Decision Tree.
- Random Forest.
- Logistic Regression.

Evaluation method:

- The merged training file is deduplicated by `(location_name, timestamp)`.
- The current demo dataset contains 504 unique location-time rows across 72 timestamps.
- Train/test splitting is grouped by timestamp, so the same simulated image context cannot appear in both sets.
- The selected Decision Tree reached macro F1 = 0.9744 on the grouped holdout set. This measures how closely it learns the transparent simulated policy, not real-world accident prevention.

Final output artifacts:

- `models/drone_decision_model.joblib`: trained best model.
- `reports/model_metrics.csv`: comparison table for report.
- `reports/classification_report.txt`: precision, recall, F1 by class.
- `reports/training_summary.json`: grouped split details and deduplication audit.
- `reports/recommendation_demo.csv`: demo-ready recommendation rows for dashboard/Power BI.
- `reports/best_slot.json`: next best golden flight window.
- `reports/backtesting_summary.csv`: baseline fixed schedule vs DSS activated KPI result.
- `reports/backtesting_daily_results.csv`: detailed daily scenario records.

## Honest limitations to present

- The current images are simulated weather-scene images and MobileNetV2 embeddings, not labeled UAV crop-surface images collected in a real field.
- `crop_condition` is currently a transparent weather-based proxy. A later version should train a separate image classifier for `HEALTHY`, `WATER_STRESS`, and `DRY_SOIL` using labeled field images.
- Backtesting compares a fixed-noon baseline with the best available DSS slot. Its percentage improvement is a simulation upper bound, not measured field performance.
- The current backtest covers 30 agricultural location-days. The specification target of 100 historical days still requires a larger historical dataset.

## Demo command

```bash
.venv/bin/python -m src.decision_model.train_decision_model
.venv/bin/python -m src.decision_model.demo_decision --location "Dong Thap"
.venv/bin/python -m src.decision_model.backtest_policy
.venv/bin/python -m src.decision_model.live_demo --location "Can Tho" --scenario lock_spray
```
