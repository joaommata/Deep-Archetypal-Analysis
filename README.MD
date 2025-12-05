# Deep Archetypal Analysis Frameworks: Comparative Evaluation

A comparative study on the effectiveness, stability, and interpretability of classical linear Archetypal Analysis (AA) versus the deep learning-based Archetypal Analysis Network (AAnet).

---

## 1. Project Description
Archetypal Analysis (AA) is a matrix decomposition technique representing data points as a convex combination of extreme data points, referred to as archetypes. By capturing boundary patterns, AA provides a geometric description that enhances interpretability, especially in high-dimensional settings. This repository evaluates the trade-offs between classical Linear AA and the non-linear AAnet.

---

## 2. Models Implemented

### 2.1 Linear Archetypal Analysis (Linear AA)
- Represents each observation as a convex combination of archetypes, and vice versa.

### 2.2 Archetypal Analysis Network (AAnet)
- Replaces linear mapping \(R = X C S\) with a non-linear autoencoder while preserving the archetypal simplex structure.

---

## 3.  Datasets and Preprocessing

| Dataset   | Type                                                                 
|----------|--------------------------------
| MNIST    | Image (Handwritten Digit '4')  
| scRNA-seq| Biological (Gene Expression) 
---