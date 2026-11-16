# =============================================================================
# Home Credit Default Risk - GCI Global 2026 in-class competition
# Final submission: output/submission_v34.csv   (public leaderboard 0.77822, stacked OOF AUC 0.77517)
#
# HOW TO RUN
#   Google Colab, T4 GPU. Folder layout:  input/train.csv, input/test.csv, input/sample_submission.csv
#   Then run:  %run v34_full.py
#   Uses ONLY input/train.csv and input/test.csv (no external data). All seeds are fixed.
#   First run trains every model from scratch (several hours). Each finished model is saved in
#   output/cache_v28/ so an interrupted run can simply be restarted and continues where it stopped.
#
# PIPELINE
#   1. Features: ratios, EXT_SOURCE statistics, implied interest rate / loan term, group-relative
#      values, out-of-fold target encodings (categories, loan products, regions), model-based
#      EXT_SOURCE imputation and repeat-customer fingerprints.
#   2. Base models (out-of-fold predictions on train, averaged predictions on test):
#      LightGBM (several variants and feature views), CatBoost, XGBoost, logistic regression,
#      TabM neural networks. Test predictions blend fold models with full-data refits.
#   3. Ensemble A: logistic-regression stacker on the 5-fold base models.
#   4. Ensemble B: the strongest base models retrained with 10 folds (extra seeds), and a stacker
#      pruned of members that do not improve out-of-fold AUC.
#   5. Final prediction: 50/50 blend (in logit space) of Ensemble A and Ensemble B.
#   Every selection step is decided by out-of-fold AUC on train only, never by the leaderboard.
#
# REFERENCE
#   TabM: Gorishniy et al. 2024, "TabM: Advancing Tabular Deep Learning with Parameter-Efficient
#   Ensembling" (TabM-mini variant), own implementation.
# =============================================================================

# %% 0. Colab setup
import os, subprocess, sys
try:
    from google.colab import drive
    drive.mount("/content/drive")
    os.chdir("/content/drive/MyDrive/GCI/Competition")       # <- your competition folder
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "catboost", "xgboost"], check=False)
except ImportError:
    pass
print("working folder:", os.getcwd())

# %% 1. Setup and data loading
import gc, warnings, random
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)

SMOKE = False          # True = quick test on a small sample (only to check the code runs)
LRX = 8 if SMOKE else 1  # learning-rate multiplier for smoke tests
N_FOLDS = 5

current_dir = Path(os.getcwd())
CACHE = current_dir / "output" / ("cache_v28_smoke" if SMOKE else "cache_v28")
CACHE.mkdir(parents=True, exist_ok=True)

try:
    from catboost.utils import get_gpu_device_count
    HAS_GPU = get_gpu_device_count() > 0
except Exception:
    HAS_GPU = False
print("GPU available:", HAS_GPU)

train = pd.read_csv(current_dir / "input" / "train.csv")
test = pd.read_csv(current_dir / "input" / "test.csv")
ss_path = current_dir / "input" / "sample_submission.csv"
sample_sub = pd.read_csv(ss_path) if ss_path.exists() else test[["SK_ID_CURR"]].copy()
if SMOKE:
    train = train.sample(20000, random_state=SEED).sort_values("SK_ID_CURR").reset_index(drop=True)
print("train:", train.shape, "test:", test.shape)

# %% 2. Feature engineering (train + test features together; TARGET is never used here)
y = train["TARGET"].values
n_train = len(train)
df = pd.concat([train.drop(columns=["TARGET"]), test], axis=0, ignore_index=True)

def has(*cols):
    return all(c in df.columns for c in cols)

df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
for c in ["ORGANIZATION_TYPE", "CODE_GENDER"]:
    df[c] = df[c].replace("XNA", np.nan)
df["DAYS_LAST_PHONE_CHANGE"] = df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan)

ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
E = df[ext]
df["EXT_MEAN"] = E.mean(axis=1); df["EXT_STD"] = E.std(axis=1)
df["EXT_MIN"] = E.min(axis=1); df["EXT_MAX"] = E.max(axis=1)
df["EXT_PROD"] = E.prod(axis=1, min_count=len(ext))
df["EXT_NAN_COUNT"] = E.isna().sum(axis=1)
w = {"EXT_SOURCE_1": 2, "EXT_SOURCE_2": 1, "EXT_SOURCE_3": 3}
num = sum(df[c].fillna(0) * w[c] for c in ext)
den = sum(df[c].notna() * w[c] for c in ext)
df["EXT_WEIGHTED"] = num / den.replace(0, np.nan)
for i in range(len(ext)):
    for j in range(i + 1, len(ext)):
        df[f"{ext[i]}_x_{ext[j]}"] = df[ext[i]] * df[ext[j]]
        df[f"{ext[i]}_NDIFF_{ext[j]}"] = (df[ext[i]] - df[ext[j]]) / (df[ext[i]] + df[ext[j]])
df["EXT_HARMONIC"] = E.notna().sum(axis=1) / (1 / E.clip(lower=1e-6)).sum(axis=1, min_count=1)

df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
df["CREDIT_TO_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
df["CREDIT_GOODS_DIFF"] = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
for c in ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
    df[f"{c}_TO_INCOME"] = df[c] / df["AMT_INCOME_TOTAL"]
df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
df["NET_INCOME_PER_PERSON"] = (df["AMT_INCOME_TOTAL"] - df["AMT_ANNUITY"]) / df["CNT_FAM_MEMBERS"]
df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (1 + df["CNT_CHILDREN"])
df["CHILDREN_RATIO"] = df["CNT_CHILDREN"] / df["CNT_FAM_MEMBERS"]

df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
for c in ["DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]:
    df[f"{c}_TO_AGE"] = df[c] / df["DAYS_BIRTH"]
df["CAR_AGE_TO_AGE"] = df["OWN_CAR_AGE"] / df["AGE_YEARS"]
df["CREDIT_PER_AGE"] = df["AMT_CREDIT"] / df["AGE_YEARS"]
df["AGE_x_EXT_MEAN"] = df["AGE_YEARS"] * df["EXT_MEAN"]
df["INCOME_PER_EMP_DAY"] = df["AMT_INCOME_TOTAL"] / (-df["DAYS_EMPLOYED"] + 1)

req = ["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON",
       "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]
df["REQ_SUM"] = df[req].sum(axis=1, min_count=1)

group_cols = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_EDUCATION_TYPE",
              "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS"]
value_cols = ["AMT_INCOME_TOTAL", "EXT_MEAN", "CREDIT_TERM", "AMT_CREDIT_TO_INCOME"]
for g in group_cols:
    for v in value_cols:
        df[f"{v}_DIV_{g}"] = df[v] / df.groupby(g)[v].transform("median")

def combo(cols):
    k = df[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + df[c].astype(str)
    return k
COMBO_DEFS = {"ORG": ["ORGANIZATION_TYPE"], "OCC": ["OCCUPATION_TYPE"],
              "INC": ["NAME_INCOME_TYPE"],
              "ORG_EDU": ["ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE"],
              "ORG_INC": ["ORGANIZATION_TYPE", "NAME_INCOME_TYPE"],
              "OCC_GEN_EDU": ["OCCUPATION_TYPE", "CODE_GENDER", "NAME_EDUCATION_TYPE"]}
combos = pd.DataFrame({n: combo(c) for n, c in COMBO_DEFS.items()})
g = combos["OCC_GEN_EDU"]
for v in ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"]:
    grp = df.groupby(g)[v]
    df[f"{v}_DIV_OCC_GEN_EDU"] = df[v] / grp.transform("median")
    df[f"{v}_Z_OCC_GEN_EDU"] = (df[v] - grp.transform("mean")) / grp.transform("std")

def implied_monthly_rate(P, A, n, iters=60):
    P = np.asarray(P, float); A = np.asarray(A, float)
    lo = np.full(P.shape, 1e-9); hi = np.full(P.shape, 0.2)
    ok = (A * n > P) & np.isfinite(P) & np.isfinite(A) & (P > 0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        pay = P * mid / (1 - (1 + mid) ** (-n))
        too_high = pay > A
        hi = np.where(too_high, mid, hi); lo = np.where(too_high, lo, mid)
    return np.where(ok, (lo + hi) / 2, np.nan)

P_, A_ = df["AMT_CREDIT"].values, df["AMT_ANNUITY"].values
revolving = (df["NAME_CONTRACT_TYPE"] == "Revolving loans").values
terms = [6, 10, 12, 18, 24, 30, 36, 42, 48, 54, 60]
R = np.column_stack([implied_monthly_rate(P_, A_, n) for n in terms])
for j, n in enumerate(terms):
    df[f"RATE_TERM_{n}"] = np.where(revolving, np.nan, R[:, j] * 12)
min_term = P_ / A_
df["MIN_TERM"] = np.where(revolving, np.nan, min_term)
feas = np.array(terms)[None, :] > min_term[:, None]
first = np.where(feas.any(1), feas.argmax(1), -1)
df["NEXT_STD_TERM"] = np.where((first >= 0) & ~revolving, np.array(terms)[first], np.nan)
df["RATE_NEXT_STD_TERM"] = np.where((first >= 0) & ~revolving,
                                    R[np.arange(len(df)), np.maximum(first, 0)] * 12, np.nan)
df["TERM_SLACK"] = df["NEXT_STD_TERM"] - df["MIN_TERM"]
df["MIN_TERM_FRAC"] = min_term - np.floor(min_term)
df["RATE_NEXT_x_CREDIT_GOODS"] = df["RATE_NEXT_STD_TERM"] * df["CREDIT_GOODS_RATIO"]

cat_cols = df.select_dtypes(include="object").columns.tolist()
for c in cat_cols:
    df[f"{c}_COUNT"] = df[c].map(df[c].value_counts()).astype(float)
    df[c] = df[c].astype("category")
df = df.replace([np.inf, -np.inf], np.nan)

features = [c for c in df.columns if c != "SK_ID_CURR"]
X = df.iloc[:n_train][features].reset_index(drop=True)
X_test = df.iloc[n_train:][features].reset_index(drop=True)
test_ids = df.iloc[n_train:]["SK_ID_CURR"].values

# %% 2b. Out-of-fold smoothed target encoding of category combinations
def oof_target_encode(k_tr, k_te, m=100):
    prior = y.mean()
    def enc_from(keys, labels):
        st = pd.DataFrame({"k": keys, "y": labels}).groupby("k")["y"].agg(["sum", "count"])
        return (st["sum"] + prior * m) / (st["count"] + m)
    oof = np.zeros(len(k_tr))
    for a, b in StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(k_tr, y):
        oof[b] = pd.Series(k_tr[b]).map(enc_from(k_tr[a], y[a])).fillna(prior).values
    te = pd.Series(k_te).map(enc_from(k_tr, y)).fillna(prior).values
    return oof, te
for name in combos.columns:
    k = combos[name].values
    X[f"TE_{name}"], X_test[f"TE_{name}"] = oof_target_encode(k[:n_train], k[n_train:])

# %% 2c. Model-based imputation of EXT_SOURCE_1/2/3 (does not use TARGET)
f_imp = CACHE / "ext_impute.npy"
allX = pd.concat([X, X_test], ignore_index=True)
base = [c for c in allX.columns if "EXT" not in c and not c.startswith("TE_")]
if f_imp.exists():
    Pimp = np.load(f_imp); print("[cache] ext_impute")
else:
    print("[TRAINING] ext_impute")
    Pimp = np.zeros((len(allX), len(ext)))
    rp = dict(objective="regression", learning_rate=0.05 * LRX, num_leaves=63, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=-1, seed=SEED)
    cc = [c for c in cat_cols if c in base]
    for j, e in enumerate(ext):
        known = allX[e].notna().values
        Xk, yk = allX.loc[known, base], allX.loc[known, e].values
        oof = np.zeros(known.sum())
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(Xk):
            m = lgb.train(rp, lgb.Dataset(Xk.iloc[a], yk[a], categorical_feature=cc), 600)
            oof[b] = m.predict(Xk.iloc[b])
        m = lgb.train(rp, lgb.Dataset(Xk, yk, categorical_feature=cc), 600)
        Pimp[known, j] = oof
        if (~known).any():
            Pimp[~known, j] = m.predict(allX.loc[~known, base])
    np.save(f_imp, Pimp)
new = pd.DataFrame(index=allX.index); filled = {}
for j, e in enumerate(ext):
    new[f"{e}_PRED"] = Pimp[:, j]
    filled[e] = allX[e].fillna(pd.Series(Pimp[:, j], index=allX.index))
    new[f"{e}_FILLED"] = filled[e]
    new[f"{e}_RESID"] = allX[e] - Pimp[:, j]
F = pd.DataFrame(filled)
new["EXT_FILLED_MEAN"] = F.mean(axis=1); new["EXT_FILLED_MIN"] = F.min(axis=1)
new["EXT_FILLED_PROD"] = F.prod(axis=1)
new["EXT_FILLED_WEIGHTED"] = sum(F[e] * w[e] for e in ext) / sum(w.values())
for c in new.columns:
    X[c] = new[c].values[:n_train]; X_test[c] = new[c].values[n_train:]

# %% 2d. Repeat-customer fingerprints (same person applying again keeps the same
#  "absolute" dates: birth minus ID-issue / registration / employment day)
allX = pd.concat([X, X_test], ignore_index=True)
b = allX["DAYS_BIRTH"].astype(float)
gnd = allX["CODE_GENDER"].astype(str)
def make_key(parts):
    ok = pd.concat(parts, axis=1).notna().all(axis=1)
    k = gnd.copy()
    for p in parts:
        k = k + "|" + p.round(0).astype("Int64").astype(str)
    return k.where(ok)
KEYS = {"K_IDREG": make_key([b - allX["DAYS_ID_PUBLISH"], b - allX["DAYS_REGISTRATION"]]),
        "K_IDEMP": make_key([b - allX["DAYS_ID_PUBLISH"], b - allX["DAYS_EMPLOYED"]]),
        "K_REGEMP": make_key([b - allX["DAYS_REGISTRATION"], b - allX["DAYS_EMPLOYED"]])}
newc = {}
for name, k in KEYS.items():
    newc[f"{name}_CNT"] = k.map(k.value_counts()).values
    t = pd.DataFrame({"k": k, "b": b}).dropna(subset=["k"]).sort_values(["k", "b"], ascending=[True, False])
    gprev = -t.groupby("k")["b"].diff(); gnext = t.groupby("k")["b"].diff(-1)
    order = t.groupby("k").cumcount() + 1
    for nm, ser in [("GAP_PREV", gprev), ("GAP_NEXT", gnext), ("ORDER", order)]:
        col = pd.Series(np.nan, index=allX.index); col.loc[ser.index] = ser.values
        newc[f"{name}_{nm}"] = col.values
    k_tr, k_te = k.iloc[:n_train].values, k.iloc[n_train:].values
    def stats(keys, labels):
        d = pd.DataFrame({"k": keys, "y": labels}).dropna(subset=["k"])
        return d.groupby("k")["y"].agg(["mean", "count"])
    m_tr = np.full(n_train, np.nan); c_tr = np.zeros(n_train)
    for a, bb in StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n_train), y):
        st = stats(k_tr[a], y[a]); kb = pd.Series(k_tr[bb])
        m_tr[bb] = kb.map(st["mean"]).values; c_tr[bb] = kb.map(st["count"]).fillna(0).values
    st = stats(k_tr, y); kt = pd.Series(k_te)
    newc[f"{name}_TGT_MEAN"] = np.concatenate([m_tr, kt.map(st["mean"]).values])
    newc[f"{name}_TGT_CNT"] = np.concatenate([c_tr, kt.map(st["count"]).fillna(0).values])
for c in newc:
    X[c] = newc[c][:n_train]; X_test[c] = newc[c][n_train:]

# %% 3. Loan-product / region target encodings and feature sets
#  FEATS   = lean set for tree models (107 features)
#  ALLCOLS = every engineered feature (used by CatBoost-ALL and the TabM networks)
RAW = [c for c in test.columns if c != "SK_ID_CURR"]
def group_of(c):
    if c in RAW: return "RAW"
    if c.startswith("K_"): return "CLIENT"
    if c.startswith("TE_"): return "TE"
    if any(t in c for t in ["_PRED", "_FILLED", "_RESID"]): return "IMPUTE"
    if c.startswith("RATE_") or c in ["MIN_TERM", "NEXT_STD_TERM", "TERM_SLACK", "MIN_TERM_FRAC"]: return "RATE"
    return "OTHER"
LEAN = [c for c in X.columns if group_of(c) in ("RAW", "RATE", "IMPUTE", "CLIENT")]

raw_all = pd.concat([train.drop(columns=["TARGET"]), test], ignore_index=True)
ratio = raw_all["AMT_CREDIT"] / raw_all["AMT_ANNUITY"]
PROD_KEYS = {
    "GOODS": raw_all.AMT_GOODS_PRICE.astype(str), "CREDIT": raw_all.AMT_CREDIT.astype(str),
    "ANNUITY": raw_all.AMT_ANNUITY.astype(str),
    "CRED_ANN": raw_all.AMT_CREDIT.astype(str) + "|" + raw_all.AMT_ANNUITY.astype(str),
    "GOODS_CRED": raw_all.AMT_GOODS_PRICE.astype(str) + "|" + raw_all.AMT_CREDIT.astype(str),
    "RATIO2": ratio.round(2).astype(str), "RATIO3": ratio.round(3).astype(str), "RATIO1": ratio.round(1).astype(str),
    "RATIO2_CTYPE": ratio.round(2).astype(str) + "|" + raw_all.NAME_CONTRACT_TYPE.astype(str),
    "ANN_GOODS3": (raw_all.AMT_ANNUITY / raw_all.AMT_GOODS_PRICE).round(4).astype(str),
    "RATIO2_INS": ratio.round(2).astype(str) + "|" + (raw_all.AMT_CREDIT > raw_all.AMT_GOODS_PRICE).astype(str)}
prior = y.mean(); M_SMOOTH = 50
def oof_te(keys):
    keys = keys.values; o = np.zeros(n_train)
    for a, bb in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(np.zeros(n_train), y):
        st = pd.DataFrame({"k": keys[:n_train][a], "y": y[a]}).groupby("k")["y"].agg(["sum", "count"])
        o[bb] = pd.Series(keys[:n_train][bb]).map((st["sum"] + prior * M_SMOOTH) / (st["count"] + M_SMOOTH)).fillna(prior).values
    st = pd.DataFrame({"k": keys[:n_train], "y": y}).groupby("k")["y"].agg(["sum", "count"])
    t = pd.Series(keys[n_train:]).map((st["sum"] + prior * M_SMOOTH) / (st["count"] + M_SMOOTH)).fillna(prior).values
    return o, t
for k, sr in PROD_KEYS.items():
    X[f"PROD_{k}_TE"], X_test[f"PROD_{k}_TE"] = oof_te(sr)
    cnt = sr.map(sr.value_counts()).values.astype(float)
    X[f"PROD_{k}_CNT"], X_test[f"PROD_{k}_CNT"] = cnt[:n_train], cnt[n_train:]
reg = raw_all.REGION_POPULATION_RELATIVE.astype(str)
reqp = (raw_all.AMT_REQ_CREDIT_BUREAU_HOUR.astype(str) + "|" + raw_all.AMT_REQ_CREDIT_BUREAU_MON.astype(str) + "|"
        + raw_all.AMT_REQ_CREDIT_BUREAU_QRT.astype(str) + "|" + raw_all.AMT_REQ_CREDIT_BUREAU_YEAR.astype(str))
REGION_KEYS = {"REGION": reg, "REGION_x_RATING": reg + "|" + raw_all.REGION_RATING_CLIENT.astype(str),
               "REGION_x_PRODUCT": reg + "|" + ratio.round(1).astype(str),
               "REQ_YEAR": raw_all.AMT_REQ_CREDIT_BUREAU_YEAR.astype(str), "REQ_PATTERN": reqp}
for k, sr in REGION_KEYS.items():
    X[f"HT_{k}_TE"], X_test[f"HT_{k}_TE"] = oof_te(sr)

FEATS = LEAN + [c for c in X.columns if c.startswith("PROD_") or c.startswith("HT_")]
CC = [c for c in cat_cols if c in FEATS]
ALLCOLS = list(X.columns)
NOPROD = [c for c in FEATS if not c.startswith("PROD_") and not c.startswith("HT_")]
V4COLS = [c for c in ALLCOLS if not c.startswith("PROD_") and not c.startswith("HT_")]   # older v4/v5 feature view
print(f"lean features: {len(FEATS)}   all features: {len(ALLCOLS)}   no-product: {len(NOPROD)}   v4 view: {len(V4COLS)}")
assert len(FEATS) == 107, "lean feature count changed - stop and check!"

# %% 4. Model library (each returns out-of-fold predictions on train + predictions on test)
def cached(name, fn):
    fo, fp = CACHE / f"{name}_oof.npy", CACHE / f"{name}_pred.npy"
    if fo.exists() and fp.exists():
        print(f"[cache] {name}"); return np.load(fo), np.load(fp)
    print(f"[TRAINING] {name}")
    o, p = fn(); np.save(fo, o); np.save(fp, p)
    print(f"   -> {name} OOF AUC {roc_auc_score(y, o):.5f}"); gc.collect(); return o, p

PC = dict(objective="binary", metric="auc", learning_rate=0.01 * LRX, verbose=-1, n_jobs=-1,
          num_leaves=42, max_depth=4, min_child_samples=47, feature_fraction=0.4198,
          bagging_fraction=0.7824, bagging_freq=1, lambda_l1=1.3826, lambda_l2=0.00867,
          min_split_gain=0.0514, cat_smooth=61, min_data_per_group=33)
PDEEP = {**PC, "num_leaves": 255, "max_depth": -1, "min_child_samples": 200,
         "feature_fraction": 0.6, "lambda_l2": 10, "learning_rate": 0.02 * LRX}
PC_ET = {**PC, "extra_trees": True}

def lgb_cv(params, seed, cols=None, k=5):
    cols = FEATS if cols is None else cols
    cc = [c for c in CC if c in cols]
    Xs, Xt = X[cols], X_test[cols]; oof = np.zeros(n_train); pred = np.zeros(len(Xt)); its = []
    for f, (tr, va) in enumerate(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(Xs, y)):
        dtr = lgb.Dataset(Xs.iloc[tr], y[tr], categorical_feature=cc)
        dva = lgb.Dataset(Xs.iloc[va], y[va], categorical_feature=cc, reference=dtr)
        m = lgb.train({**params, "seed": seed + f}, dtr, 20000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = m.predict(Xs.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(Xt, num_iteration=m.best_iteration) / k; its.append(m.best_iteration)
        print(f"  fold {f+1}: {roc_auc_score(y[va], oof[va]):.5f} (iter {m.best_iteration})")
    full = lgb.train({**params, "seed": seed}, lgb.Dataset(Xs, y, categorical_feature=cc), int(np.mean(its) * 1.15))
    return oof, 0.5 * pred + 0.5 * full.predict(Xt)

RANK_COLS = ["EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_FILLED_MEAN", "RATE_NEXT_STD_TERM", "AMT_ANNUITY"]
def lgb_rank(seed, k=5):
    Xb, Xtb = X[RANK_COLS].copy(), X_test[RANK_COLS].copy()
    allr = pd.concat([Xb, Xtb], ignore_index=True).rank(pct=True)
    for c in RANK_COLS:
        X[c] = allr[c].values[:n_train]; X_test[c] = allr[c].values[n_train:]
    try:
        return lgb_cv(PC, seed, k=k)
    finally:
        for c in RANK_COLS:
            X[c] = Xb[c].values; X_test[c] = Xtb[c].values

def logreg_cv():
    from scipy import sparse
    from sklearn.preprocessing import OneHotEncoder
    allF = pd.concat([X[FEATS], X_test[FEATS]], ignore_index=True); B = pd.DataFrame(index=allF.index)
    for c in FEATS:
        col = allF[c]
        B[c] = col.astype(str) if (str(col.dtype) == "category" or col.nunique() <= 30) \
            else pd.qcut(col.rank(method="first"), 20, labels=False).astype(str)
    Mx = sparse.csr_matrix(OneHotEncoder(handle_unknown="ignore", min_frequency=50).fit_transform(B))
    Mtr, Mte = Mx[:n_train], Mx[n_train:]; oof = np.zeros(n_train); pred = np.zeros(len(X_test))
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(Mtr, y):
        m = LogisticRegression(C=0.05, solver="liblinear", max_iter=300).fit(Mtr[tr], y[tr])
        oof[va] = m.predict_proba(Mtr[va])[:, 1]; pred += m.predict_proba(Mte)[:, 1] / 5
    full = LogisticRegression(C=0.05, solver="liblinear", max_iter=300).fit(Mtr, y)
    return oof, 0.5 * pred + 0.5 * full.predict_proba(Mte)[:, 1]

def cat_cv(cseed, cols, k=5):
    from catboost import CatBoostClassifier, Pool
    cc = [c for c in cols if str(X[c].dtype) == "category"]
    Xc, Xt = X[cols].copy(), X_test[cols].copy()
    for c in cc:
        Xc[c] = Xc[c].astype(object).fillna("NA").astype(str); Xt[c] = Xt[c].astype(object).fillna("NA").astype(str)
    oof = np.zeros(n_train); pred = np.zeros(len(Xt))
    for f, (tr, va) in enumerate(StratifiedKFold(n_splits=k, shuffle=True, random_state=cseed).split(Xc, y)):
        prm = dict(iterations=10000, learning_rate=(0.03 if HAS_GPU else 0.05) * LRX, depth=6, l2_leaf_reg=5,
                   eval_metric="AUC", random_seed=cseed + f, od_type="Iter", od_wait=200, verbose=0)
        prm.update(dict(task_type="GPU", devices="0") if HAS_GPU else dict(thread_count=-1))
        m = CatBoostClassifier(**prm)
        m.fit(Pool(Xc.iloc[tr], y[tr], cat_features=cc), eval_set=Pool(Xc.iloc[va], y[va], cat_features=cc),
              use_best_model=True)
        oof[va] = m.predict_proba(Xc.iloc[va])[:, 1]; pred += m.predict_proba(Xt)[:, 1] / k
        print(f"  CatBoost fold {f+1}: {roc_auc_score(y[va], oof[va]):.5f} (iter {m.get_best_iteration()})")
    return oof, pred

def xgb_cv(seed):
    import xgboost as xgb
    Xs, Xt = X[FEATS], X_test[FEATS]; oof = np.zeros(n_train); pred = np.zeros(len(Xt)); its = []
    prm = dict(n_estimators=15000, learning_rate=0.015 * LRX, max_depth=5, subsample=0.8,
               colsample_bytree=0.5, min_child_weight=5, reg_lambda=5, tree_method="hist",
               enable_categorical=True, max_cat_to_onehot=1, eval_metric="auc",
               random_state=seed, device="cuda" if HAS_GPU else "cpu")
    for k, (tr, va) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(Xs, y)):
        m = xgb.XGBClassifier(**prm, early_stopping_rounds=200)
        m.fit(Xs.iloc[tr], y[tr], eval_set=[(Xs.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(Xs.iloc[va])[:, 1]; pred += m.predict_proba(Xt)[:, 1] / 5
        its.append(m.best_iteration + 1)
        print(f"  XGB fold {k+1}: {roc_auc_score(y[va], oof[va]):.5f} (trees {m.best_iteration + 1})")
    full = xgb.XGBClassifier(**{**prm, "n_estimators": int(np.mean(its) * 1.15)})
    full.fit(Xs, y, verbose=False)
    return oof, 0.5 * pred + 0.5 * full.predict_proba(Xt)[:, 1]

# ---- TabM neural network (needs PyTorch, preinstalled on Colab) ----
try:
    import torch, torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available -> TabM members skipped")

if HAS_TORCH:
    TORCH_DEV = "cuda" if torch.cuda.is_available() else "cpu"
    class TabM(nn.Module):
        def __init__(s, d_num, cards, k=8, width=256, depth=3, drop=0.15):
            super().__init__()
            s.embs = nn.ModuleList([nn.Embedding(c, min(16, (c + 1) // 2)) for c in cards])
            D = d_num + sum(e.embedding_dim for e in s.embs)
            s.r = nn.Parameter(torch.randint(0, 2, (k, D)).float() * 2 - 1)
            layers = []
            for i in range(depth):
                layers += [nn.Linear(D if i == 0 else width, width), nn.ReLU(), nn.Dropout(drop)]
            s.mlp = nn.Sequential(*layers)
            s.head_w = nn.Parameter(torch.randn(k, width) * width ** -0.5)
            s.head_b = nn.Parameter(torch.zeros(k))
        def forward(s, xn, xc):
            h = torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(s.embs)], 1)
            h = s.mlp(h[:, None, :] * s.r[None])
            return (h * s.head_w[None]).sum(-1) + s.head_b

    def tabm_fit_predict(Xtr, Ctr, ytr, Xva, Cva, yva, Xte, Cte, cards, seed,
                         epochs=30, patience=5, bs=1024, lr=1e-3, wd=3e-4):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if TORCH_DEV == "cuda": torch.cuda.manual_seed_all(seed)
        net = TabM(Xtr.shape[1], cards).to(TORCH_DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        T = lambda a, dt: torch.as_tensor(a, dtype=dt, device=TORCH_DEV)
        xtr, ctr, yt = T(Xtr, torch.float32), T(Ctr, torch.long), T(ytr, torch.float32)
        lossf = nn.BCEWithLogitsLoss()
        def pred(Xn, Cn):
            net.eval(); out = []
            with torch.no_grad():
                for i in range(0, len(Xn), 4096):
                    out.append(torch.sigmoid(net(T(Xn[i:i+4096], torch.float32),
                                                 T(Cn[i:i+4096], torch.long))).mean(1).cpu().numpy())
            return np.concatenate(out)
        g_ = torch.Generator(device=TORCH_DEV); g_.manual_seed(seed)
        best, bad, bv, bt = 0, 0, None, None
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(xtr), device=TORCH_DEV, generator=g_)
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]; out = net(xtr[j], ctr[j])
                loss = lossf(out, yt[j][:, None].expand_as(out))
                opt.zero_grad(); loss.backward(); opt.step()
            pv = pred(Xva, Cva); a = roc_auc_score(yva, pv)
            if a > best + 1e-5: best, bad, bv, bt = a, 0, pv, pred(Xte, Cte)
            else:
                bad += 1
                if bad >= patience: break
        return best, bv, bt, ep + 1

    def tabm_cv(seed, cols, k=5):
        cc = [c for c in cols if str(X[c].dtype) == "category"]; numc = [c for c in cols if c not in cc]
        allF = pd.concat([X[cols], X_test[cols]], ignore_index=True)
        C = np.stack([allF[c].cat.codes.values.astype(np.int64) + 1 for c in cc], 1)
        cards = [len(allF[c].cat.categories) + 1 for c in cc]
        Ctr_all, Cte = C[:n_train], C[n_train:]
        oof = np.zeros(n_train); pred = np.zeros(len(X_test))
        for f, (tr, va) in enumerate(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n_train), y)):
            Ntr = X.iloc[tr][numc].astype(float).replace([np.inf, -np.inf], np.nan)
            qt = QuantileTransformer(n_quantiles=500, output_distribution="normal", subsample=10**6, random_state=0).fit(Ntr)
            miss = [c for c in numc if Ntr[c].isna().mean() > 0.01]
            def tf(D):
                D = D[numc].astype(float).replace([np.inf, -np.inf], np.nan)
                return np.hstack([np.nan_to_num(qt.transform(D), nan=0.0).clip(-5, 5), D[miss].isna().values]).astype(np.float32)
            best, bv, bt, eps = tabm_fit_predict(tf(X.iloc[tr]), Ctr_all[tr], y[tr], tf(X.iloc[va]), Ctr_all[va], y[va],
                                                 tf(X_test), Cte, cards, seed + f, epochs=4 if SMOKE else 30)
            oof[va] = bv; pred += bt / k
            print(f"  TabM fold {f+1}: {best:.5f} (epochs {eps})")
        return oof, pred

# %% 5. Base models, 5-fold (Ensemble A members)
S = (lambda full, smoke: smoke if SMOKE else full)
def avg(tag, seeds, fn):
    res = [cached(f"{tag}_{s_}", lambda: fn(s_)) for s_ in seeds]
    return np.mean([o for o, _ in res], 0), np.mean([p for _, p in res], 0)

O, T = {}, {}
O["lgbc"], T["lgbc"] = avg("lgbc", S([42, 7, 2026, 11, 99, 123, 456, 789, 1001, 2024], [42]), lambda s_: lgb_cv(PC, s_))
O["lgbc_et"], T["lgbc_et"] = avg("lgbc_et", S([42, 7, 2026, 11, 99], [42]), lambda s_: lgb_cv(PC_ET, s_))
O["deep"], T["deep"] = avg("deep", [99], lambda s_: lgb_cv(PDEEP, s_))
O["lgb_noprod"], T["lgb_noprod"] = avg("noprod", S([303, 404, 505], [303]), lambda s_: lgb_cv(PC, s_, NOPROD))
# same LightGBM on a wider feature view (no product/region encodings) -> diversity for the stacker
O["lgb_v4"], T["lgb_v4"] = avg("lgb_v4", S([111, 222], [111]), lambda s_: lgb_cv(PC, s_, V4COLS))
O["lgbc_rank"], T["lgbc_rank"] = avg("lgbc_rank", [4242], lgb_rank)
O["logreg"], T["logreg"] = cached("logreg", logreg_cv)
O["cat"], T["cat"] = avg("cat", S([42, 1234, 777, 31, 555], [42]), lambda s_: cat_cv(s_, FEATS))
O["cat_all"], T["cat_all"] = avg("cat_all", S([42, 1234, 777], [42]), lambda s_: cat_cv(s_, ALLCOLS))
O["xgb"], T["xgb"] = avg("xgb", S([42, 2026], [42]), xgb_cv)
if HAS_TORCH:
    O["tabm"], T["tabm"] = avg("tabm", S([42, 7, 2026], [42]), lambda s_: tabm_cv(s_, FEATS))
    O["tabm_all"], T["tabm_all"] = avg("tabm_all", S([42, 7, 2026], [42]), lambda s_: tabm_cv(s_, ALLCOLS))

# %% 5b. Full-data refits for CatBoost and TabM
#  Fold models only saw 80% of train. Each refit trains once on 100% of train and is averaged 50/50
#  with the fold models for the TEST predictions only (OOF predictions and stacker weights are unchanged).
#  Refit lengths come from the 5-fold runs: CatBoost best iteration ~1,900 (x1.15 for 25% more data,
#  at GPU lr 0.03), TabM best epoch ~11 (x1.15).
CAT_REFIT_ITERS, TABM_REFIT_EPOCHS = 2200, 13
def cached_refit(name, fn):
    f = CACHE / f"refit_{name}.npy"
    if f.exists():
        print(f"[cache] refit_{name}"); return np.load(f)
    print(f"[TRAINING] refit_{name}")
    p = fn(); np.save(f, p); gc.collect(); return p

def cat_refit(cseed, cols):
    from catboost import CatBoostClassifier, Pool
    cc = [c for c in cols if str(X[c].dtype) == "category"]
    Xc, Xt = X[cols].copy(), X_test[cols].copy()
    for c in cc:
        Xc[c] = Xc[c].astype(object).fillna("NA").astype(str); Xt[c] = Xt[c].astype(object).fillna("NA").astype(str)
    lr = (0.03 if HAS_GPU else 0.05) * LRX
    prm = dict(iterations=max(50, int(CAT_REFIT_ITERS * 0.03 / lr)), learning_rate=lr, depth=6, l2_leaf_reg=5,
               random_seed=cseed, verbose=0)
    prm.update(dict(task_type="GPU", devices="0") if HAS_GPU else dict(thread_count=-1))
    m = CatBoostClassifier(**prm); m.fit(Pool(Xc, y, cat_features=cc))
    return m.predict_proba(Pool(Xt, cat_features=cc))[:, 1]

CAT_SEEDS, CAT_ALL_SEEDS = S([42, 1234, 777, 31, 555], [42]), S([42, 1234, 777], [42])
r = np.mean([cached_refit(f"cat_{s_}", lambda: cat_refit(s_, FEATS)) for s_ in CAT_SEEDS], 0)
T["cat"] = 0.5 * T["cat"] + 0.5 * r
r = np.mean([cached_refit(f"cat_all_{s_}", lambda: cat_refit(s_, ALLCOLS)) for s_ in CAT_ALL_SEEDS], 0)
T["cat_all"] = 0.5 * T["cat_all"] + 0.5 * r

if HAS_TORCH:
    def tabm_refit(seed, cols, epochs):
        cc = [c for c in cols if str(X[c].dtype) == "category"]; numc = [c for c in cols if c not in cc]
        allF = pd.concat([X[cols], X_test[cols]], ignore_index=True)
        C = np.stack([allF[c].cat.codes.values.astype(np.int64) + 1 for c in cc], 1)
        cards = [len(allF[c].cat.categories) + 1 for c in cc]
        Ntr = X[numc].astype(float).replace([np.inf, -np.inf], np.nan)
        qt = QuantileTransformer(n_quantiles=500, output_distribution="normal", subsample=10**6, random_state=0).fit(Ntr)
        miss = [c for c in numc if Ntr[c].isna().mean() > 0.01]
        def tf(D):
            D = D[numc].astype(float).replace([np.inf, -np.inf], np.nan)
            return np.hstack([np.nan_to_num(qt.transform(D), nan=0.0).clip(-5, 5), D[miss].isna().values]).astype(np.float32)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        if TORCH_DEV == "cuda": torch.cuda.manual_seed_all(seed)
        Tt = lambda a, dt: torch.as_tensor(a, dtype=dt, device=TORCH_DEV)
        xtr, ctr, yt = Tt(tf(X), torch.float32), Tt(C[:n_train], torch.long), Tt(y, torch.float32)
        xte, cte = tf(X_test), C[n_train:]
        net = TabM(xtr.shape[1], cards).to(TORCH_DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=3e-4); lossf = nn.BCEWithLogitsLoss()
        g_ = torch.Generator(device=TORCH_DEV); g_.manual_seed(seed)
        for ep in range(epochs):
            net.train(); perm = torch.randperm(len(xtr), device=TORCH_DEV, generator=g_)
            for i in range(0, len(perm), 1024):
                j = perm[i:i + 1024]; out = net(xtr[j], ctr[j])
                loss = lossf(out, yt[j][:, None].expand_as(out))
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(xte), 4096):
                out.append(torch.sigmoid(net(Tt(xte[i:i+4096], torch.float32), Tt(cte[i:i+4096], torch.long))).mean(1).cpu().numpy())
        return np.concatenate(out)
    E = 3 if SMOKE else TABM_REFIT_EPOCHS
    TS = S([42, 7, 2026], [42])
    r = np.mean([cached_refit(f"tabm_{s_}", lambda: tabm_refit(s_, FEATS, E)) for s_ in TS], 0)
    T["tabm"] = 0.5 * T["tabm"] + 0.5 * r
    r = np.mean([cached_refit(f"tabm_all_{s_}", lambda: tabm_refit(s_, ALLCOLS, E)) for s_ in TS], 0)
    T["tabm_all"] = 0.5 * T["tabm_all"] + 0.5 * r

# %% 5c. 10-fold versions of the strongest members (Ensemble B members)
#  Each fold model learns from 90% of train instead of 80%. The stacker then chooses, by out-of-fold
#  AUC on two different stacker splits, whether to keep the 5-fold members, swap in, or add the 10-fold ones.
K10 = 10
O10, T10 = {}, {}
O10["lgbc"], T10["lgbc"] = avg("lgbc10", S([42, 7, 2026], [42]), lambda s_: lgb_cv(PC, s_, k=K10))
O10["lgb_v4"], T10["lgb_v4"] = avg("lgb_v4_10", S([111, 222, 333, 444], [111]), lambda s_: lgb_cv(PC, s_, V4COLS, k=K10))
O10["lgbc_rank"], T10["lgbc_rank"] = avg("lgbc_rank10", S([4242, 1313, 2424, 3535], [4242]), lambda s_: lgb_rank(s_, k=K10))
if HAS_TORCH:
    TS10 = S([42, 7, 2026, 11, 99, 123], [42])
    O10["tabm_all"], t_ = avg("tabm_all10", TS10, lambda s_: tabm_cv(s_, ALLCOLS, k=K10))
    T10["tabm_all"] = 0.5 * t_ + 0.5 * np.mean([cached_refit(f"tabm_all_{s_}", lambda: tabm_refit(s_, ALLCOLS, E)) for s_ in TS10], 0)
C10 = S([42, 1234], [42])
O10["cat"], t_ = avg("cat10", C10, lambda s_: cat_cv(s_, FEATS, k=K10))
T10["cat"] = 0.5 * t_ + 0.5 * np.mean([cached_refit(f"cat_{s_}", lambda: cat_refit(s_, FEATS)) for s_ in C10], 0)
O10["cat_all"], t_ = avg("cat_all10", C10, lambda s_: cat_cv(s_, ALLCOLS, k=K10))
T10["cat_all"] = 0.5 * t_ + 0.5 * np.mean([cached_refit(f"cat_all_{s_}", lambda: cat_refit(s_, ALLCOLS)) for s_ in C10], 0)
print("\n10-fold vs 5-fold member OOF AUC:")
for k_ in O10:
    print(f"  {k_:10s} 5-fold {roc_auc_score(y, O[k_]):.5f}   10-fold {roc_auc_score(y, O10[k_]):.5f}   ({roc_auc_score(y, O10[k_]) - roc_auc_score(y, O[k_]):+.5f})")

lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
def stack_oof(Od, seed):
    Zs = np.column_stack([lg(Od[k_]) for k_ in Od]); s_o = np.zeros(n_train)
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(Zs, y):
        s_o[va] = LogisticRegression(C=1.0, max_iter=2000).fit(Zs[tr], y[tr]).predict_proba(Zs[va])[:, 1]
    return roc_auc_score(y, s_o)
VAR = {"keep 5-fold": (O, T),
       "swap to 10-fold": ({**O, **O10}, {**T, **T10}),
       "add 10-fold too": ({**O, **{k_ + "_10": v for k_, v in O10.items()}}, {**T, **{k_ + "_10": v for k_, v in T10.items()}})}
print("\nstacker OOF AUC (two different stacker splits):")
sc = {}
for nm, (Od, _) in VAR.items():
    a1, a2 = stack_oof(Od, 123), stack_oof(Od, 7); sc[nm] = (a1, a2)
    print(f"  {nm:20s} {a1:.5f}  {a2:.5f}")
base1, base2 = sc["keep 5-fold"]
ok = [nm for nm in VAR if nm != "keep 5-fold" and sc[nm][0] > base1 + 0.0001 and sc[nm][1] > base2 + 0.0001]
CHOICE = max(ok, key=lambda nm: sum(sc[nm])) if ok else "keep 5-fold"
print(f">>> chosen: {CHOICE}")
O, T = VAR[CHOICE]

# %% 5d. Stacker tuning: greedy member pruning and regularisation strength
#  A change is kept only if out-of-fold AUC improves on BOTH stacker splits.
def stack_c(Od, keys, seed, C):
    Zs = np.column_stack([lg(Od[k_]) for k_ in keys]); s_o = np.zeros(n_train)
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(Zs, y):
        s_o[va] = LogisticRegression(C=C, max_iter=2000).fit(Zs[tr], y[tr]).predict_proba(Zs[va])[:, 1]
    return roc_auc_score(y, s_o)
def score(keys, C):
    return stack_c(O, keys, 123, C), stack_c(O, keys, 7, C)
KEYS = list(O); STK_C = 1.0
base = score(KEYS, STK_C); cur = base
print(f"\nstacker before tuning: {base[0]:.5f}  {base[1]:.5f}   ({len(KEYS)} members, C=1)")
MIN_GAIN = 0.00002
while len(KEYS) > 3:                                   # greedy pruning
    trials = {k_: score([m for m in KEYS if m != k_], STK_C) for k_ in KEYS}
    k_best = max(trials, key=lambda k_: sum(trials[k_]))
    a1, a2 = trials[k_best]
    if a1 > cur[0] + MIN_GAIN and a2 > cur[1] + MIN_GAIN:
        KEYS.remove(k_best); cur = (a1, a2)
        print(f"  drop {k_best:12s} -> {a1:.5f}  {a2:.5f}")
    else:
        break
for C_ in [0.03, 0.1, 0.3, 3.0]:                        # regularisation strength
    a1, a2 = score(KEYS, C_)
    print(f"  C={C_:<5}         -> {a1:.5f}  {a2:.5f}")
    if a1 > cur[0] + MIN_GAIN and a2 > cur[1] + MIN_GAIN:
        cur, STK_C = (a1, a2), C_
IMPROVED = cur[0] > base[0] + 0.00005 and cur[1] > base[1] + 0.00005
if not IMPROVED:
    KEYS, STK_C, cur = list(O), 1.0, base
print(f">>> tuned stacker: {cur[0]:.5f}  {cur[1]:.5f}   members kept {len(KEYS)}/{len(O)}, C={STK_C}   "
      + ("(tuning kept)" if IMPROVED else "(no clear gain, tuning discarded)"))
O = {k_: O[k_] for k_ in KEYS}; T = {k_: T[k_] for k_ in KEYS}

print("\nMember OOF AUCs:")
for k in O:
    print(f"  {k:12s} {roc_auc_score(y, O[k]):.5f}")

# %% 6. Ensemble B: logistic-regression stacker on logits (nested CV for an honest score)
lg = lambda v: np.log(np.clip(v, 1e-6, 1 - 1e-6) / (1 - np.clip(v, 1e-6, 1 - 1e-6)))
names = list(O)
Z = np.column_stack([lg(O[k]) for k in names]); Zt = np.column_stack([lg(T[k]) for k in names])
oof_stack = np.zeros(n_train)
for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(Z, y):
    oof_stack[va] = LogisticRegression(C=STK_C, max_iter=2000).fit(Z[tr], y[tr]).predict_proba(Z[va])[:, 1]
stk = LogisticRegression(C=STK_C, max_iter=2000).fit(Z, y)
pred = stk.predict_proba(Zt)[:, 1]
auc = roc_auc_score(y, oof_stack)
print("\nstacker weights:", {k: round(float(c), 3) for k, c in zip(names, stk.coef_[0])})

# %% 7. Ensemble A (5-fold members) and the final 50/50 blend
#  Both stackers use the same split, so the blended out-of-fold AUC is an honest estimate.
O29, T29 = VAR["keep 5-fold"]
Z29 = np.column_stack([lg(O29[k_]) for k_ in O29]); Zt29 = np.column_stack([lg(T29[k_]) for k_ in O29])
oof29 = np.zeros(n_train)
for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(Z29, y):
    oof29[va] = LogisticRegression(C=1.0, max_iter=2000).fit(Z29[tr], y[tr]).predict_proba(Z29[va])[:, 1]
pred29 = LogisticRegression(C=1.0, max_iter=2000).fit(Z29, y).predict_proba(Zt29)[:, 1]
sig = lambda z: 1 / (1 + np.exp(-z))
oof_bl = sig(0.5 * lg(oof29) + 0.5 * lg(oof_stack)); pred = sig(0.5 * lg(pred29) + 0.5 * lg(pred))
print(f"\nOOF AUC  Ensemble A {roc_auc_score(y, oof29):.5f} | Ensemble B {auc:.5f} | 50/50 blend {roc_auc_score(y, oof_bl):.5f}")
auc = roc_auc_score(y, oof_bl)

sub = sample_sub[["SK_ID_CURR"]].merge(pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": pred}),
                                       on="SK_ID_CURR", how="left")
assert sub["TARGET"].notna().all(), "Some IDs did not get a prediction!"
out = current_dir / "output" / ("submission_v34_SMOKE.csv" if SMOKE else "submission_v34.csv")
sub.to_csv(out, index=False)

print("\n" + "=" * 70)
print(f"Final blended OOF AUC: {auc:.5f}   saved: {out}")
print("=" * 70)
