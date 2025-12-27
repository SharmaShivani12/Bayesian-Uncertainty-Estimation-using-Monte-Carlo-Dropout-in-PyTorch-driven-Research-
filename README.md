
# Bayesian Uncertainty Estimation using Monte Carlo Dropout in PyTorch 🧠🔎

🧠 Overview

This project demonstrates Bayesian uncertainty estimation using Monte Carlo Dropout in a neural network trained on the Breast Cancer Wisconsin Diagnostic Dataset.
Instead of returning a single prediction, the model estimates both prediction and uncertainty, which is essential for aerospace, reliability-critical ML systems, anomaly detection, and healthcare diagnostics.

Traditional deep learning returns a number.
Probabilistic deep learning returns a confidence level behind that number.

This work showcases how dropout at inference can approximate Bayesian Neural Networks, enabling models to express when they're confident or uncertain — a key component in trustworthy & interpretable AI.

**Kaggle Notebook** : https://www.kaggle.com/code/shivanisharma1297/bayesian-uncertainty-mini-study/

## 📊 Dataset

Breast Cancer Wisconsin Diagnostic Dataset

Binary classification: Malignant vs Benign

30 input features per sample

Well-suited for uncertainty research & calibration studies

📎 Source: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

---

## 🚀 Project Summary

This project implements a small neural network with **dropout active at inference time**, enabling Bayesian approximation by sampling multiple forward passes.  
From these samples, we compute:

- **Mean prediction** → model output
- **Standard deviation** → uncertainty estimate (confidence)

This helps identify when the model is unsure — essential for **probabilistic deep learning and uncertainty quantification**.
| Method                 | Description                             |
| ---------------------- | --------------------------------------- |
| **Architecture**       | 2-layer MLP with Dropout                |
| **Training Objective** | Binary cross-entropy                    |
| **Bayesian Mechanism** | Dropout enabled during inference        |
| **Uncertainty Source** | Variance across multiple forward passes |


---

## 📂 Files

| File | Description |
|------|-------------|
| `Bayesian_Uncertainty_Mini_Study.py` | Main script demonstrating Bayesian inference with dropout |

---

## 🔧 Installation & Setup

```bash
# Clone repo
git clone <your_repo_link>

cd Bayesian-Uncertainty-Mini-Study

# Install required packages
pip install torch numpy
```

---

## ▶ Run the Script

```bash
python Bayesian_Uncertainty_Mini_Study.py
```

You will see output similar to:

```
Training Completed
Prediction Mean: [0.53]
Uncertainty (Std): [0.12]
```

Higher uncertainty → model is less confident.  
Lower uncertainty → more reliable prediction.

---
##🔍 Result Summary

<img width="317" height="97" alt="image" src="https://github.com/user-attachments/assets/98e0cef9-c136-4d51-881b-3c8c5c2479a0" />


<img width="588" height="463" alt="image" src="https://github.com/user-attachments/assets/3cc68e98-d12c-4672-b962-4482cdd34a0e" />


<img width="589" height="452" alt="image" src="https://github.com/user-attachments/assets/4dd2e71a-0399-47d5-b943-d5c27e50739a" />



---

## ✨ Author

**Shivani Sharma**  
AI/ML Engineer | Research & Deep Learning | Bayesian/Uncertainty Learning  
GitHub: https://github.com/SharmaShivani12  


