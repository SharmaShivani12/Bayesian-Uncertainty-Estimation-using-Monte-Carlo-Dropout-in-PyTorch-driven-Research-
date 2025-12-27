
# Bayesian Uncertainty Mini-Study 🧠🔎

This mini project demonstrates **Bayesian uncertainty estimation using Monte Carlo Dropout in PyTorch**.  
The goal is to show how neural networks can express **predictive confidence**, not just predictions — a valuable concept for **reliable AI systems such as aerospace, medical diagnostics, safety-critical ML, and research applications**.

---

## 🚀 Project Summary

This project implements a small neural network with **dropout active at inference time**, enabling Bayesian approximation by sampling multiple forward passes.  
From these samples, we compute:

- **Mean prediction** → model output
- **Standard deviation** → uncertainty estimate (confidence)

This helps identify when the model is unsure — essential for **probabilistic deep learning and uncertainty quantification**.

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

## 📌 Concepts Demonstrated

- Monte Carlo Dropout
- Bayesian Neural Networks (approximation)
- Uncertainty Quantification
- PyTorch experimentation workflow
- Reproducible ML mini research setup

---

## 🧱 Potential Extensions

Feel free to extend this project:

- Train on real datasets instead of random tensors  
- Add visualization for uncertainty distribution  
- Compare deterministic vs Bayesian behavior  
- Apply to aerospace, fault-detection, or safety-critical ML tasks  
- Add calibration metrics (ECE, reliability diagrams)  

---

## ✨ Author

**Shivani Sharma**  
AI/ML Engineer | Research & Deep Learning | Bayesian/Uncertainty Learning  
GitHub: https://github.com/SharmaShivani12  


