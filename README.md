# Kaggle: Road Accident Risk Prediction (Playground S5E10)

This repository contains my solution for the Kaggle Playground Series (S5E10) competition on road accident risk prediction. My final approach achieved a score of **0.05551**. (Rank 619/4083 on private leaderboard, 602/4083 on public leaderboard).

## Overview

My strategy involved two main stages:
1.  **L1 Model Training:** Using AutoGluon (`best_quality`) with extensive feature engineering (base features + interactions).
2.  **L2 Stacking:** Training a `RidgeCV` meta-model on the Out-of-Fold (OOF) predictions from the top 10 L1 models. This stacker model was key to finding complex negative weights and improving the final score.

## Notebooks

* **`eda.ipynb`**: Exploratory Data Analysis, feature exploration, and initial analysis of target distributions.
* **`model_pipeline.ipynb`**: The main notebook containing:
    1.  Feature engineering.
    2.  Training the L1 AutoGluon models.
    3.  Extracting OOF predictions.
    4.  Training the L2 `RidgeCV` stacker.
    5.  Error analysis of the final stacker.
