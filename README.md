# 🎾 Tennis Ranking & Match Prediction using Perron–Frobenius and Machine Learning

**Author**: Paul-Emile Marcus  
**Environment**: Python 3 (Google Colab)  
**Keywords**: Graph theory, Linear algebra, Machine learning, Sports analytics, Random Forest, Perron–Frobenius theorem, Elo rating  

---

## 🧠 Project Overview

This project explores how **mathematical modeling** and **machine learning** can be used to analyze and predict tennis performance.  
Originally developed as a **TIPE project (academic research in French preparatory classes)**, it was later restructured to serve as a **data science portfolio project**.

The work is divided into two main parts:

1. **Ranking system based on the Perron–Frobenius theorem**  
   → Construction of a player ranking matrix derived from match results and relative strength indicators.  
2. **Match outcome prediction using a Random Forest classifier**  
   → Use of the custom ranking as a feature to train and evaluate predictive models.

---

## 🧩 Mathematical Background

### 1. From Elo to Relative Strength
The Elo model defines the probability of player *A* defeating player *B* as:

\[
p(D) = \frac{1}{1 + 10^{-\frac{D}{400}}}
\]

To capture more nuances, several parameters were combined:
- **Elo differential** (`PELO`)
- **Head-to-head differential** (`PH2H`)
- **Win ratio differential** (`PRATIO`)

A weighted sum of these probabilities defines the **relative strength matrix** between players.

---

### 2. Perron–Frobenius Ranking
The matrix of pairwise win probabilities is treated as a **stochastic matrix**.  
By iteratively applying it to an initial uniform vector, we obtain convergence (via the **Perron–Frobenius theorem**) toward the **dominant eigenvector**, which represents each player’s stationary strength.

This yields a **ranking independent from the ATP algorithm**, built solely from match data and probabilistic relationships.

---

## ⚙️ Implementation

### Structure
├── TIPE_spé_classement.ipynb   # Ranking computation using Perron–Frobenius
├── TIPE_spé_ML.ipynb           # Machine learning model (Random Forest)
├── data/                       # Match datasets (not included)
└── README.md                   # Project documentation


