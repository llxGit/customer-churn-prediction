# Customer Churn Prediction

## 🇬🇧 English

### About the project

Machine learning project for predicting customer churn in a telecommunications company

The model analyzes customer information such as contract type, monthly charges, subscription duration and services to predict whether a customer is likely to leave

### Dataset

IBM Telco Customer Churn

- 7,043 original records
- 7,032 records after cleaning
- 19 input features
- Target: `Churn`, where 0 = Stay and 1 = Leave

### Models

Four models were compared using 5-fold cross-validation

| Model | CV F1 |
|---|---:|
| Logistic Regression | 0.5960 |
| Gradient Boosting | 0.5873 |
| Random Forest | 0.5712 |
| Baseline | 0.0000 |

Logistic Regression was selected for hyperparameter tuning

### Final results

Best parameters: `C=1`, `class_weight='balanced'`

Best cross-validation F1: **0.6322**

| Metric | Test result |
|---|---:|
| Accuracy | 72.57% |
| Precision | 49.01% |
| Recall | 79.68% |
| F1 | 0.6069 |

The final model correctly identified 298 of 374 customers who left, with 310 false positives

### Technologies

Python, pandas, scikit-learn, Matplotlib, Jupyter, joblib

### How to run

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python -m src.train
```

Evaluate the model

```bash
python -m src.evaluate
```

Run a prediction

```bash
python -m src.predict
```

Compare models

```bash
python -m src.model_compare
```

Tune hyperparameters

```bash
python -m src.tune
```

---

## 🇷🇺 Русский

### О проекте

Проект машинного обучения для прогнозирования оттока клиентов телекоммуникационной компании

Модель анализирует данные о клиентах, включая тип контракта, ежемесячные платежи, срок обслуживания и подключённые услуги, чтобы определить вероятность ухода клиента

### Датасет

IBM Telco Customer Churn

- 7 043 исходные записи
- 7 032 записи после очистки
- 19 входных признаков
- Целевая переменная: `Churn`, где 0 = Останется и 1 = Уйдёт

### Модели

Сравнение четырёх моделей с использованием 5-кратной кросс-валидации

| Модель | CV F1 |
|---|---:|
| Logistic Regression | 0.5960 |
| Gradient Boosting | 0.5873 |
| Random Forest | 0.5712 |
| Baseline | 0.0000 |

Для настройки гиперпараметров выбрана Logistic Regression

### Результаты

Лучшие параметры: `C=1`, `class_weight='balanced'`

Лучший F1 на кросс-валидации: **0.6322**

| Метрика | Тестовый результат |
|---|---:|
| Accuracy | 72.57% |
| Precision | 49.01% |
| Recall | 79.68% |
| F1 | 0.6069 |

Модель правильно определила 298 из 374 клиентов, которые действительно ушли, при этом ошибочно отметила 310 клиентов как потенциально уходящих

### Технологии

Python, pandas, scikit-learn, Matplotlib, Jupyter, joblib

### Запуск

Установка библиотек

```bash
pip install -r requirements.txt
```

Обучение модели

```bash
python -m src.train
```

Оценка качества

```bash
python -m src.evaluate
```

Прогноз для нового клиента

```bash
python -m src.predict
```

Сравнение моделей

```bash
python -m src.model_compare
```

Подбор гиперпараметров

```bash
python -m src.tune
```