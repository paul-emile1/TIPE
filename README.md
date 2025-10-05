# Tennis Ranking & Match Prediction using Perron–Frobenius and Machine Learning

**Author**: Paul-Emile Marcus  
**Environment**: Python 3 (Google Colab)  
**Keywords**: Graph theory, Linear algebra, Machine learning, Sports analytics, Random Forest, Perron–Frobenius theorem, Elo rating  

---

## Project Overview

This project explores how **mathematical modeling** and **machine learning** can be used to analyze and predict tennis performance.  
Originally developed as a **TIPE project (academic research in French preparatory classes)**, it was later restructured to serve as a **data science portfolio project**.

The work is divided into two main parts:

1. **Ranking system based on the Perron–Frobenius theorem**  
   → Construction of a player ranking matrix derived from match results and relative strength indicators.  
2. **Match outcome prediction using a Random Forest classifier**  
   → Use of the custom ranking as a feature to train and evaluate predictive models.

---

## Mathematical Background

### 1. From Elo to Relative Strength

The Elo model defines the probability of player *A* defeating player *B* as:

p(D) = 1 / (1 + 10^(-D / 400))

To capture more nuances, several parameters were combined:
- **Elo differential** (`PELO`)
- **Head-to-head differential** (`PH2H`)
- **Win ratio differential** (`PRATIO`)

A weighted sum of these probabilities defines the **relative strength matrix** between players.

---

### 2. The Perron–Frobenius Theorem

At the heart of this project lies a classical result from linear algebra — the **Perron–Frobenius theorem**, which ensures the convergence of a positive, primitive stochastic matrix toward a unique dominant eigenvector.

**Theorem (simplified):**

> Let `A` be a positive and primitive square matrix of size `n`.  
> Then there exists a unique positive eigenvalue `r = ρ(A)` (the spectral radius) such that:
>
> - `r` is simple (algebraic multiplicity 1)  
> - There exists a unique positive vector `x₀` satisfying `A x₀ = r x₀` and `‖x₀‖₁ = 1`.

In this project:
- `A` represents the **transition matrix** describing the probability that player *i* beats player *j*.
- The **dominant eigenvector** of `A` gives a stationary distribution of "strength" across all players.

---

## Implementation
```
TIPE-tennis-ranking/
│
├── TIPE_spé_classement.ipynb # Ranking computation using Perron–Frobenius
├── TIPE_spé_ML.ipynb # Machine learning model (Random Forest)
├── data/ # Match datasets (not included)
└── README.md # Project documentation
```

### Environment

- **Language**: Python 3  
- **Platform**: Google Colab  
- **Main Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - *(plus standard libraries: `datetime`, `math`, `random`, `copy`, `time`)*

---

## Data Preprocessing

Data preprocessing includes:
- Merging ATP match results from multiple tournaments.  
- Cleaning inconsistent player names (via Levenshtein distance).  
- Building head-to-head matrices and ratio statistics.  
- Normalizing values for matrix construction.  

The resulting **transition matrix** represents the estimated probability that player *i* beats player *j*.

---

## Ranking Algorithm

1. Construct the transition matrix `A` of pairwise probabilities.  
2. Initialize a positive vector `Y0` such that `||Y0||₁ = 1`.  
3. Iterate:  
   \[
   Y_{n+1} = A \cdot Y_n
   \]
4. Convergence to the stationary vector `Y∞` gives the final ranking.

This process is guaranteed to converge by the **Perron–Frobenius theorem** for positive and primitive matrices.

---

## Machine Learning Model

Once the custom ranking was computed, it was integrated as a feature in a **binary classification** task predicting match outcomes.

- **Model**: `RandomForestClassifier` (from `scikit-learn`)  
- **Target**: whether the better-ranked player wins  
- **Features**: player rankings (custom & ATP), match metadata  
- **Evaluation metrics**: accuracy, confusion matrix, probability calibration  

---

## Usage

Open the notebooks in Google Colab or locally:

```bash
# Ranking computation
TIPE_spé_classement.ipynb

# Model training and evaluation
TIPE_spé_ML.ipynb

### Project Structure

