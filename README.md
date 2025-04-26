# 🧠 Training a Machine Learning Model with Non-Differentiable Loss Function (Hinge Loss) Using Subgradient Descent

This project is a final assignment focused on implementing a Support Vector Machine (SVM) from scratch using the **Hinge Loss** function — a non-differentiable loss — and optimizing it with **Gradient Descent** and **Subgradient Descent** techniques.

---

## 📌 Project Overview

This repository contains a full solution for training a machine learning model using a non-differentiable loss function. It satisfies the academic requirements by:
- Choosing **Hinge Loss**
- Using the [Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
- Building an **SVM classifier from scratch**
- Applying two optimization strategies:
  - Gradient Descent
  - Subgradient Descent
- Tracking and reporting loss and accuracy across epochs

---

## 📊 Dataset

**Banknote Authentication Dataset**  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

### Features:
- Variance of Wavelet Transformed Image
- Skewness of Wavelet Transformed Image
- Curtosis of Wavelet Transformed Image
- Entropy of Image

### Target:
- Class (0: Forged, 1: Genuine)

---

## 🛠️ Project Structure

```
📁 data/
    └── data_banknote_authentication.txt   # Raw dataset
📄 classifier creation.ipynb               # Model training and loss visualization
📄 testing model.ipynb                     # Model testing and performance
📄 bombordinocrocodilo.py                  # Core implementation script
📄 README.md                               # Project documentation
```

---

## 📈 Optimization Methods

### 1. Gradient Descent
Used for functions where gradient is defined and smooth.

### 2. Subgradient Descent
Designed to handle **non-differentiable loss functions** like Hinge Loss.

---

## 📚 Training Metrics

During training, the following were recorded:
- 🔺 Loss per epoch
- ✅ Accuracy on training set
- ✅ Accuracy on validation set

These metrics help evaluate convergence and model performance.

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/omarabdelrazikk/training-ML-model-with-Non-Differentiable-Loss-Function-Hinge-loss-Using-Sub-gradient-Descent
   cd training-ML-model-with-Non-Differentiable-Loss-Function-Hinge-loss-Using-Sub-gradient-Descent
   ```

2. Open and run the notebooks:
   - `classifier creation.ipynb` for training
   - `testing model.ipynb` for evaluation

> Make sure to have Jupyter Notebook and Python 3.x installed with `numpy`, `pandas`, and `matplotlib`.

---

## 🚀 Results

- The model achieved high accuracy on both training and testing sets.
- Subgradient Descent proved effective in optimizing non-differentiable Hinge Loss.

---

## 📬 Contact

For questions or contributions, feel free to reach out via GitHub or open an issue!