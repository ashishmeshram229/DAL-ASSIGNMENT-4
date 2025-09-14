# DA5401 A4: GMM-Based Synthetic Sampling for Imbalanced Data

**Student Name:** Ashish Meshram  
**Roll No:** DA25M016  
**Institute:** IIT Madras  

---

## Overview

This assignment focuses on tackling the class imbalance problem in the context of fraud detection using a Gaussian Mixture Model (GMM) for synthetic data generation. The dataset used is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

The primary goal is to augment the minority (fraudulent) class to create a balanced training dataset and analyze the impact on model performance compared to a baseline model trained on imbalanced data.

---

## Contents

1. **Part A: Baseline Model and Data Analysis**
    - Data loading and exploration
    - Analysis of class distribution and imbalance
    - Logistic Regression model training on imbalanced data
    - Baseline evaluation using Precision, Recall, and F1-score for the minority class

2. **Part B: Gaussian Mixture Model (GMM) for Synthetic Sampling**
    - Theoretical explanation of GMM vs SMOTE
    - Implementation of GMM for minority class oversampling
    - Determination of optimal number of components using AIC/BIC
    - Synthetic data generation and combination with original data
    - Rebalancing using Clustering-Based Undersampling (CBU)

3. **Part C: Performance Evaluation and Conclusion**
    - Training Logistic Regression on GMM-balanced dataset
    - Evaluation on original imbalanced test set
    - Comparative analysis of baseline vs GMM-balanced model
    - Final recommendation on using GMM for synthetic sampling

---

## Highlights of the Approach

- **GMM-based oversampling:** Captures complex distributions and sub-groups in the minority class better than traditional methods like SMOTE.
- **Clustering-Based Undersampling (CBU):** Ensures the majority class is reduced without losing important distributional characteristics.
- **Performance Metrics:** Precision, Recall, and F1-score for the minority class were prioritized over accuracy to reflect the real impact on fraud detection.

---

## Results Summary

| Model | Precision (Fraud) | Recall (Fraud) | F1-score (Fraud) |
|-------|-----------------|----------------|----------------|
| Baseline Logistic Regression | 0.725 | 0.674 | 0.698 |
| GMM Oversampled | 0.908 | 0.918 | 0.918 |
| GMM + CBU Balanced | 0.919 | 0.918 | 0.918 |

**Key Observations:**
- GMM-based oversampling significantly improved the model's ability to detect fraudulent transactions.
- Combining GMM with CBU further enhanced balance without degrading performance on the majority class.
- Accuracy alone is not a good metric due to extreme imbalance; minority-class-specific metrics are more informative.

---

## Conclusion

The assignment demonstrates that GMM-based synthetic sampling is highly effective for handling imbalanced datasets, particularly when the minority class exhibits complex structures. It improves the model's detection of fraud while maintaining robustness on the majority class.

**Recommendation:** For fraud detection tasks with imbalanced data, GMM-based oversampling combined with careful undersampling of the majority class (CBU) is recommended over simpler methods like SMOTE.

---

## How to Run

1. Open the `DA5401_A4.ipynb` Jupyter Notebook.
2. Ensure `creditcard.csv` is in the same directory or update the path in the notebook.
3. Install dependencies if not already installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
