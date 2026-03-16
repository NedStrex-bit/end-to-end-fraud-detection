# Fraud Detection System

Fraud Detection System — это end-to-end ML-проект для выявления мошеннических транзакций. На вход система получает данные на уровне транзакций, на выходе возвращает вероятность fraud и бинарное решение на основе настраиваемого threshold.

## Структура проекта

```text
.
├── api/
├── artifacts/
│   ├── models/
│   ├── plots/
│   └── reports/
├── configs/
├── data/
│   ├── processed/
│   └── raw/
├── src/
│   ├── data/
│   ├── features/
│   ├── inference/
│   ├── models/
│   └── utils/
└── tests/
```

## Окружение

- Python 3.11+
- Установка зависимостей:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Размещение датасета

Поместите исходный датасет в [data/raw](./data/raw). Поддерживаемые форматы:

- `.csv`
- `.parquet`

Примеры:

- `data/raw/transactions.csv`
- `data/raw/fraud_dataset.parquet`

## Запуск Data Audit

Запуск audit для конкретного файла:

```bash
python -m src.data.run_data_audit --file transactions.csv
```

Или автоматический выбор первого подходящего файла из `data/raw`:

```bash
python -m src.data.run_data_audit
```

Сохраняемые артефакты:

- Markdown-отчёт: [artifacts/reports/data_audit_report.md](./artifacts/reports/data_audit_report.md)
- JSON-сводка: [artifacts/reports/data_audit_summary.json](./artifacts/reports/data_audit_summary.json)

## Запуск EDA

Запуск EDA для конкретного датасета:

```bash
python -m src.data.run_eda --file transactions.csv
```

Или использование первого поддерживаемого файла из `data/raw`:

```bash
python -m src.data.run_eda
```

Артефакты EDA:

- Markdown-отчёт: [artifacts/reports/eda_report.md](./artifacts/reports/eda_report.md)
- Каталог графиков: [artifacts/plots](./artifacts/plots)

Основные графики, которые строятся при наличии нужных колонок:

- распределение target
- распределение суммы для `amount` / `transaction_amount` / `value`
- fraud rate по категориальным группам
- сравнение percentiles суммы для fraud и non-fraud
- fraud rate по часу
- fraud rate по дню недели
- heatmap корреляций числовых признаков

Если target, timestamp или amount-подобная колонка отсутствуют, pipeline не падает, а фиксирует пропущенный шаг в отчёте.

## Запуск Feature Engineering

Запуск feature pipeline для конкретного датасета:

```bash
python -m src.features.run_feature_pipeline --file transactions.csv
```

Или автоматический выбор первого подходящего файла из `data/raw`:

```bash
python -m src.features.run_feature_pipeline
```

Результаты:

- Обработанные split'ы в [data/processed](./data/processed)
- Метаданные признаков в [artifacts/reports/feature_list.json](./artifacts/reports/feature_list.json)
- Отчёт по признакам в [artifacts/reports/feature_report.md](./artifacts/reports/feature_report.md)

Поведение pipeline:

- Если есть timestamp, используется time-based split.
- Если timestamp отсутствует, pipeline переходит на stratified/random split и фиксирует это ограничение.
- Поведенческие user-level признаки строятся только по прошлым транзакциям.
- Encoders и imputers обучаются только на train, затем применяются к valid/test.
- Raw ID-поля и raw timestamp не попадают напрямую в model-ready датасет.

## Запуск Baseline Model

Обучение и оценка baseline-модели logistic regression:

```bash
python -m src.models.train_baseline
```

Результаты:

- Сохранённая модель: [artifacts/models/baseline_logistic_regression.joblib](./artifacts/models/baseline_logistic_regression.joblib)
- Метрики: [artifacts/reports/baseline_metrics.json](./artifacts/reports/baseline_metrics.json)
- Markdown-отчёт: [artifacts/reports/baseline_report.md](./artifacts/reports/baseline_report.md)
- Кривые и графики в [artifacts/plots](./artifacts/plots)

Замечания по baseline:

- Используется `LogisticRegression` с `class_weight='balanced'`.
- Основной фокус на `PR-AUC`, `Recall`, `Precision` и `F1`, а не на accuracy.
- `PR-AUC` особенно важен для fraud detection, потому что fraud-класс обычно редкий.
- Logistic regression используется как стартовая точка; для табличных данных ожидается, что более сильная модель покажет лучший результат.

## Запуск Main Model

Обучение основной модели и сравнение с baseline:

```bash
python -m src.models.train_model
```

Результаты:

- Основная модель: [artifacts/models/main_hist_gradient_boosting.joblib](./artifacts/models/main_hist_gradient_boosting.joblib)
- Метрики основной модели: [artifacts/reports/main_model_metrics.json](./artifacts/reports/main_model_metrics.json)
- Таблица сравнения моделей: [artifacts/reports/model_comparison.csv](./artifacts/reports/model_comparison.csv)
- Отчёт по основной модели: [artifacts/reports/model_report.md](./artifacts/reports/model_report.md)

Замечания по основной модели:

- Текущий основной кандидат — `HistGradientBoostingClassifier`.
- Он хорошо подходит к текущему encoded tabular pipeline и сильнее линейной logistic regression.
- Сравнение всё так же ориентировано на `PR-AUC`, `Recall`, `Precision` и `F1`; accuracy вторична.
- Рабочий threshold подбирается отдельно с учётом стоимости false positives и false negatives.

## Запуск Threshold Tuning

Запуск threshold analysis на validation split:

```bash
python -m src.models.run_threshold_analysis --split valid
```

Можно переопределить бизнес-стоимости ошибок:

```bash
python -m src.models.run_threshold_analysis --split valid --cost-fn 100 --cost-fp 5
```

Результаты:

- Таблица threshold'ов: [artifacts/reports/threshold_metrics.csv](./artifacts/reports/threshold_metrics.csv)
- Отчёт по threshold tuning: [artifacts/reports/threshold_report.md](./artifacts/reports/threshold_report.md)
- Графики в [artifacts/plots](./artifacts/plots)

Замечания по threshold tuning:

- Threshold — это часть продуктового решения, а не только техническая деталь.
- Более низкий threshold обычно увеличивает recall и число false positives.
- Более высокий threshold обычно увеличивает precision и уменьшает число лишних алертов.
- Рекомендуемый threshold выбирается из balanced-режима по минимальной бизнес-стоимости ошибок при заданных FP/FN cost.
- На текущем sample dataset рекомендованный threshold равен `0.95`, но это временное значение, потому что в validation split нет fraud-примеров.

## Запуск Explainability

Запуск explainability и error analysis на validation split:

```bash
python -m src.models.run_explainability --split valid
```

Результаты:

- Explainability report: [artifacts/reports/explainability_report.md](./artifacts/reports/explainability_report.md)
- Error analysis report: [artifacts/reports/error_analysis_report.md](./artifacts/reports/error_analysis_report.md)
- Графики важности признаков и сравнения ошибок в [artifacts/plots](./artifacts/plots)

Замечания по explainability:

- `shap` используется только если разумно доступен в окружении.
- Если `shap` недоступен, pipeline автоматически переходит на permutation importance и явно фиксирует это ограничение.
- Текущий sample dataset слишком мал для сильных выводов по explainability и error patterns, поэтому отчёты честно помечают ограничения данных.

## Локальный запуск API

Запуск API:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Проверка health:

```bash
curl http://127.0.0.1:8000/health
```

Пример запроса на предсказание:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_demo_001",
    "user_id": "user_123",
    "transaction_time": "2026-03-01T10:15:00",
    "amount": 125.50,
    "currency": "USD",
    "merchant": "Store_A"
  }'
```

Ожидаемая форма ответа:

```json
{
  "fraud_probability": 0.5,
  "fraud_label": 0,
  "threshold": 0.95
}
```

## Запуск тестов

```bash
pytest tests/test_api.py
```

## Запуск через Docker

Сборка образа:

```bash
docker build -t fraud-detection-system .
```

Запуск контейнера:

```bash
docker run --rm -p 8000:8000 fraud-detection-system
```

## Запуск Advanced Experiments

Запуск опционального anomaly detection эксперимента:

```bash
python -m src.models.run_anomaly_experiments
```

Результаты:

- Отчёт по anomaly detection: [artifacts/reports/anomaly_report.md](./artifacts/reports/anomaly_report.md)
- Метрики anomaly detection: [artifacts/reports/anomaly_metrics.csv](./artifacts/reports/anomaly_metrics.csv)
- Графики распределения score в [artifacts/plots](./artifacts/plots)

Замечания по advanced experiments:

- Текущий anomaly experiment использует `IsolationForest`.
- Он опционален и не заменяет основной supervised pipeline.
- Он может быть полезен как дополнительный сигнал для новых или слабо размеченных fraud-паттернов.
- В текущей версии проекта supervised modeling остаётся основным production-подходом.

## Финальные результаты

- Реализован полный end-to-end ML workflow: от raw data до обученной fraud-модели.
- Feature engineering сделан воспроизводимым и leakage-aware.
- Baseline и основная tabular model обучаются и сравниваются.
- Threshold tuning учитывает бизнес-стоимость ошибок.
- Генерируются отчёты по explainability и error analysis.
- FastAPI inference service предоставляет `/health` и `/predict`.

## Текущие ограничения

- Встроенный sample dataset слишком мал для содержательного выбора модели, threshold и explainability-выводов.
- Инференс для одной транзакции не имеет внешнего исторического state, поэтому user-history features строятся только из данных, доступных в самом запросе.
- Выбранный threshold на sample dataset — это временное значение, которое нужно переоценивать на реалистичном holdout-наборе.

## Scope Sprint 9

- Минимальная воспроизводимая структура ML-проекта
- Загрузка CSV и Parquet
- Первичный data audit
- Сохранение audit-артефактов
- Воспроизводимый EDA pipeline с графиками и markdown-отчётами
- Воспроизводимый feature engineering pipeline с train/valid/test
- Baseline model с сохранением метрик, графиков и артефакта модели
- Основная tabular model с умеренным tuning и сравнением с baseline
- Threshold tuning с business-cost analysis и рабочим threshold
- Explainability и error analysis с сохранением графиков и отчётов
- FastAPI inference service, Docker packaging, smoke tests и финальный summary report
- Опциональный anomaly detection эксперимент и сравнение с supervised model
