# 🧬 Breast Cancer Diagnosis: High-Recall ML Classification Pipeline

![Status](https://img.shields.io/badge/Status-Work--In--Progress-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

An end-to-end machine learning research project focused on building a high-sensitivity diagnostic tool for breast cancer classification. Developed at **Izmir Institute of Technology (IYTE)**, this project prioritizes **Recall (Sensitivity)** to minimize the risk of missing malignant cases.

---

## 🎯 Clinical Problem Statement
In oncology, a **False Negative** (classifying a malignant tumor as benign) is significantly more dangerous than a False Positive. Therefore, our primary optimization metric is **Recall for the Malignant class**, aiming to provide a safety-first diagnostic support system.

## 📊 Experimental Results & Benchmarking
We benchmarked 5 distinct machine learning architectures. The **SVM** model, optimized via **GridSearchCV**, outperformed other models in detecting malignant instances.

| Model Architecture | Recall (Malignant) | Accuracy (Overall) | Strategy |
| :--- | :---: | :---: | :--- |
| **SVM (Champion)** | **0.82** | 0.76 | High-penalty C=100, Linear Kernel |
| Logistic Regression | 0.79 | 0.77 | L2 Regularization, Balanced Weights |
| XGBoost | 0.76 | 0.75 | Scale_pos_weight optimization |
| Voting Ensemble | 0.74 | 0.77 | Soft-Voting (LR + RF + XGB + SVM) |
| Random Forest | 0.70 | 0.72 | Gini Impurity, 100 Estimators |

---

## 🛠️ The "Data Amputation" Phase (Outlier Handling)
The raw dataset contained extreme noise that sabotaged initial model training. Some features exhibited astronomical scales that required aggressive preprocessing:

- **The Energy Anomaly:** `original_firstorder_Energy` values reached levels of 10^11, dwarfing other features.
- **The Coarseness Scale:** `original_ngtdm_Coarseness` displayed values up to 10^6.

### **Preprocessing Solutions Implemented:**
1. **IQR-Based Detection:** Systematic identification of anomalies beyond Q3 + 1.5 * IQR.
2. **Logarithmic Scaling:** Applied `np.log1p` transformations to neutralize the scale dominance of high-magnitude features.
3. **Robust Scaling:** Implemented scaling techniques focused on medians and quantiles to prevent outlier-driven bias.

---

## 🏗️ Repository Structure
```text
/Brest_cancer
  ├── /data                # Preprocessed and Log-Scaled datasets
  ├── /models              # GridSearch scripts & serialized .joblib models
  ├── /notebooks           # EDA, Outlier Analysis, and Feature Engineering
  ├── /reports/figures     # Confusion Matrices and ROC-AUC Curves
  ├── requirements.txt     # Project dependencies
  └── main.py              # Model inference and evaluation script
```

---

## 🚀 Getting Started

### **1. Prerequisites**

Ensure you have the following installed on your system:

- **Python 3.10+** (Tested on 3.13)
- `pip` (Python package manager)

### **2. Installation**

```bash
# Clone the repository
git clone https://github.com/Kadirydx/Brest_cancer.git
cd Brest_cancer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt
```

### **3. Usage**

To run the full pipeline including preprocessing and evaluation:

```bash
python main.py
```

---

## 📅 Roadmap & Future Enhancements (Work in Progress)

- [x] Benchmarking of core classification models.
- [x] Outlier handling and Log-Scale stabilization.
- [x] Hyperparameter optimization with GridSearchCV.
- [ ] **Data Augmentation:** Integrating **SMOTE** (Synthetic Minority Over-sampling) to address class imbalance.
- [ ] **Feature Selection:** Utilizing **Recursive Feature Elimination (RFE)** to remove low-variance noise.
- [ ] **Deployment:** Building a **FastAPI** wrapper for real-time diagnostic predictions.
- [ ] **Visualization:** Developing a **Streamlit** dashboard for interactive data exploration.

---

## Contact & Collaboration

**Kadir Yurdakul** - Molecular Biology & Genetics Student @ IYTE

- **GitHub:** [@Kadirydx](https://github.com/Kadirydx)
- **LinkedIn:** [Kadir Yurdakul](https://www.linkedin.com/in/kadir-yurdakul-798166211/)
- **Project Topic:** Machine Learning in Computational Biology

---

*Disclaimer: This tool is intended for research and educational purposes only and should not be used as a primary diagnostic tool in clinical settings.*