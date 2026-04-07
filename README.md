# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using supervised learning techniques. The model is trained on a real-world dataset and achieves high accuracy in identifying fraudulent activity while minimizing false positives.

---

## 📌 Project Overview

Credit card fraud is a major financial problem worldwide. This project builds a binary classification model to distinguish between **legitimate** and **fraudulent** transactions based on anonymized transaction features.

---

## 📂 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── app.ipynb             # Main Jupyter Notebook (EDA, training, evaluation)
├── fraud_model.pkl       # Trained ML model (serialized)
├── scaler.pkl            # Fitted StandardScaler (serialized)
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignored files (e.g., large CSV dataset)
└── README.md             # Project documentation
```

---

## 📊 Dataset

The dataset used is the **Credit Card Fraud Detection** dataset from Kaggle.

- 📥 **Download here:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- After downloading, place `creditcard.csv` in the root project folder.

| Feature | Description |
|--------|-------------|
| `Time` | Seconds elapsed between this and the first transaction |
| `V1–V28` | PCA-transformed anonymized features |
| `Amount` | Transaction amount |
| `Class` | 0 = Legitimate, 1 = Fraudulent |

> ⚠️ The dataset is highly imbalanced — only ~0.17% of transactions are fraudulent.

---

## 🧠 ML Workflow

1. **Exploratory Data Analysis (EDA)** — Class distribution, feature correlations
2. **Preprocessing** — Feature scaling using `StandardScaler`
3. **Handling Class Imbalance** — Undersampling / Oversampling (SMOTE)
4. **Model Training** — Classification algorithm (e.g., Logistic Regression / Random Forest)
5. **Evaluation** — Confusion matrix, Precision, Recall, F1-Score, ROC-AUC

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~99% |
| Precision | High |
| Recall | High |
| ROC-AUC | ~0.97+ |

> Exact metrics are available inside `app.ipynb`.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Kajal805-M/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.

### 4. Run the notebook
```bash
jupyter notebook app.ipynb
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellowgreen?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightblue?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Array-blue?logo=numpy)

- **Python 3.8+**
- **Pandas & NumPy** — Data manipulation
- **Scikit-learn** — Model training & evaluation
- **Matplotlib & Seaborn** — Data visualization
- **Joblib / Pickle** — Model serialization

---

## 📁 Saved Artifacts

| File | Description |
|------|-------------|
| `fraud_model.pkl` | Trained classification model |
| `scaler.pkl` | StandardScaler fitted on training data |

These can be loaded directly for inference without retraining:

```python
import pickle

with open("fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 👩‍💻 Author

**Kajal** — [GitHub Profile](https://github.com/Kajal805-M)

> ⭐ If you found this project helpful, please consider giving it a star!
