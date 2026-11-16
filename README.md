# Home Credit Default Risk: Stacked Ensemble

Solution to the **GCI World 2026 in-class competition** (Matsuo Lab, University of Tokyo GCI program), based on the Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) problem.

**Final rank: 78th of 5,500+ participants** (public leaderboard ROC-AUC 0.77822).

## Problem

Predict the probability that a loan applicant will **default** (fail to make a required payment) from their application data, so a lender can rank applicants by risk.

- **Data:** 171,202 training customers and 61,500 test customers, 32 features (income, loan amounts, age, employment history, external credit scores, etc.)
- **Target:** `TARGET` = 1 for payment difficulties, 0 otherwise (about 8% positives, so the classes are imbalanced)
- **Metric:** ROC-AUC
- **Rules:** only the provided train/test data, no external data

> The dataset is **not included** in this repository. It belongs to the competition organizers.

## Approach

**1. Feature engineering** (applied to train and test features together; the target is never used here)
- Ratios: credit/income, annuity/income, credit/goods price, income per family member
- EXT_SOURCE statistics: mean, min, max, std, product, weighted mean, pairwise interactions
- Implied **monthly interest rate and loan term**, solved from credit amount and annuity
- Values relative to the median of the customer's occupation, organization, education and income group
- **Out-of-fold target encoding** of category combinations, loan-product codes and regions (no leakage)
- **Model-based imputation** of missing EXT_SOURCE scores using LightGBM regressors
- **Repeat-customer fingerprints:** date combinations that stay constant if the same person applies again

**2. Base models** (out-of-fold predictions on train, averaged predictions on test)
- LightGBM: several variants (standard, extra-trees, deep, rank-transformed inputs, different feature views)
- CatBoost, XGBoost, logistic regression on binned features
- **TabM** neural networks (parameter-efficient ensemble MLP)
- Test predictions average the fold models with a model refit on 100% of the training data

**3. Ensembling**
- Ensemble A: logistic-regression stacker on the 5-fold base models
- Ensemble B: the strongest models retrained with **10 folds** and extra seeds; the stacker is pruned of members that don't improve out-of-fold AUC
- Final prediction: a 50/50 blend (in logit space) of Ensembles A and B

Every design choice was accepted only if it improved **cross-validated AUC on two different splits**. Leaderboard scores were not used for selection, because the public leaderboard is too noisy at this level.

## Results

| Model | ROC-AUC |
|---|---|
| Course tutorial baseline | 0.732 (public leaderboard) |
| Single LightGBM (out-of-fold) | ~0.774 |
| **Final ensemble (out-of-fold, 171k customers)** | **0.77517** |
| **Final ensemble (public leaderboard)** | **0.77822** |

**Final rank: 78th of 5,500+ participants** on the competition leaderboard.

Ideas tested that did **not** help, and were left out: DART boosting, pseudo-labeling on test data, a tree-based meta-stacker, a slower learning rate, and extra features (passport-renewal timing, number roundness, value frequencies, missingness patterns).

## How to run

1. Open Google Colab and select a **T4 GPU** runtime.
2. Create this folder layout (for example in Google Drive) and update the path in section 0 of the script:
   ```
   your_folder/
   ├── home_credit_default_risk.py
   └── input/
       ├── train.csv
       ├── test.csv
       └── sample_submission.csv
   ```
3. Run:
   ```python
   %run home_credit_default_risk.py
   ```
4. The output is written to `output/submission_v34.csv`.

The first run trains every model from scratch and takes several hours. Each finished model is cached in `output/cache_v28/`, so an interrupted run continues where it stopped. All random seeds are fixed for reproducibility.

## Tech stack

Python · pandas · NumPy · scikit-learn · LightGBM · CatBoost · XGBoost · PyTorch · Google Colab

## References

- Gorishniy et al., 2024. *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling* (TabM-mini variant, own implementation)
- Kaggle competition: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)

## Acknowledgements

Developed with the help of an AI coding assistant (Claude) for writing code and running experiments.
