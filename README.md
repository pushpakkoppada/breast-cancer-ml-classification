# Week 4 – Machine Learning Model Development and Evaluation

## Project
**Breast Cancer Classification using Machine Learning**

This project develops and evaluates two supervised classification models using the Wisconsin Diagnostic Breast Cancer dataset available through scikit-learn.

### Models
- Logistic Regression with feature scaling
- Random Forest Classifier

### Best model
**Logistic Regression**

### Test-set performance
| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.9825 | 0.9474 |
| Precision | 0.9861 | 0.9583 |
| Recall | 0.9861 | 0.9583 |
| F1-Score | 0.9861 | 0.9583 |
| ROC-AUC | 0.9954 | 0.9937 |

## Dataset
The project uses the Wisconsin Diagnostic Breast Cancer dataset distributed with scikit-learn. It contains 569 observations and 30 numerical input features computed from digitized images of breast mass cell nuclei.

## Run
```bash
pip install -r requirements.txt
python src_model.py
```

## Repository structure
- `data/` – dataset used by the report
- `visualizations/` – generated charts
- `src_model.py` – model training and evaluation
- `model_metrics.csv` – test metrics
- `requirements.txt` – Python dependencies

## Academic note
This is an educational machine-learning project. The dataset and model outputs are used for learning and evaluation and are **not** intended for clinical diagnosis or medical decision-making.
