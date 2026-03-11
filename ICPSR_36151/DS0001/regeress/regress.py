# ======================================================
# IHDS HEALTH EXPENDITURE REGRESSION PIPELINE
# ======================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA



# ======================================================
# 1) LOAD DATA
# ======================================================
data_path = "../states_datasets/data_subset_Uttar Pradesh 09.csv"

df = pd.read_csv(data_path, low_memory=False)

# ======================================================
# 1) VARIABLE LISTS
# ======================================================

Major_binary_health = [
    "MB3","MB4","MB5","MB6","MB7","MB8","MB9","MB10","MB11",
    "MB12","MB13","MB14","MB15","MB16","MB17"
]

Short_Binary_health = [
    "SM4","SM5","SM6","SM7","SM8"
]

short_cost_var = [
    "SM18","SM19","SM20","SM21","SM22"   # fixed typo
]

Major_cost_vars = [
    "MB25","MB26","MB27","MB28","MB29"
]

age_var = "RO5"
gender_var = "RO3"
# education_vars = ["ED2","ED3","ED4"]

# ======================================================
# 2) CLEANING FUNCTIONS
# ======================================================

# Convert "Yes 1"/"No 0" → 1/0
def safe_binary(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.extract(r'(\d)$')[0]
                .astype(float)
            )


# Convert numeric safely
def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# ======================================================
# 3) APPLY CLEANING
# ======================================================

safe_binary(df, Major_binary_health + Short_Binary_health + [gender_var])

safe_numeric(
    df,
    short_cost_var
    + Major_cost_vars
    + [age_var]
)

# ======================================================
# 4) CREATE TOTAL COST VARIABLES
# ======================================================

df["short_total_cost"] = df[short_cost_var].sum(axis=1, skipna=True)
df["major_total_cost"] = df[Major_cost_vars].sum(axis=1, skipna=True)

# log version (recommended)
df["log_short_cost"] = np.log1p(df["short_total_cost"])
df["log_major_cost"] = np.log1p(df["major_total_cost"])

# ======================================================
# 5) REGRESSION FUNCTION
# ======================================================
def run_regression(df, y_var, x_vars, title="Model"):

    cols = [y_var] + x_vars
    data = df[cols].copy()

    # ------------------------------------------------
    # 1. Force numeric (VERY IMPORTANT)
    # ------------------------------------------------
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    # ------------------------------------------------
    # 2. Drop only rows where Y missing
    #    (do NOT drop for all Xs)
    # ------------------------------------------------
    data = data[data[y_var].notna()]

    # fill missing X with 0 (means "no illness / no edu")
    data[x_vars] = data[x_vars].fillna(0)

    print(f"\n{title}")
    print("Rows used:", len(data))

    if len(data) == 0:
        raise ValueError("Still zero rows → check cost variable")

    y = data[y_var]
    X = sm.add_constant(data[x_vars])

    model = sm.OLS(y, X).fit(cov_type="HC3")

    print(model.summary())
    return model


# ======================================================
# 6) RUN REGRESSIONS
# ======================================================

controls = [age_var, gender_var]

# ------------------------------------------------------
# SHORT ILLNESS → SHORT COST
# ------------------------------------------------------

model_short = run_regression(
    df,
    y_var="short_total_cost",
    x_vars=Short_Binary_health + controls,
    title="SHORT COST ~ SHORT ILLNESSES"
)

# log version (better)
model_short_log = run_regression(
    df,
    y_var="log_short_cost",
    x_vars=Short_Binary_health + controls,
    title="LOG SHORT COST ~ SHORT ILLNESSES (preferred)"
)


# ------------------------------------------------------
# MAJOR ILLNESS → MAJOR COST
# ------------------------------------------------------

model_major = run_regression(
    df,
    y_var="major_total_cost",
    x_vars=Major_binary_health + controls,
    title="MAJOR COST ~ MAJOR ILLNESSES"
)

# log version (better)
model_major_log = run_regression(
    df,
    y_var="log_major_cost",
    x_vars=Major_binary_health + controls,
    title="LOG MAJOR COST ~ MAJOR ILLNESSES (preferred)"
)


# ======================================================
# 7) OPTIONAL: SAVE RESULTS TO CSV
# ======================================================

def save_coefficients(model, filename):
    model.summary2().tables[1].to_csv(filename)


save_coefficients(model_short_log, "short_cost_regression.csv")
save_coefficients(model_major_log, "major_cost_regression.csv")

print("\nSaved regression tables to CSV.")



import os

# ======================================================
# CREATE FOLDER
# ======================================================

SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# ======================================================
# 1) COEFFICIENT PLOT (SAVED)
# ======================================================

def plot_coefficients(model, title, filename):

    params = model.params.drop("const")
    conf = model.conf_int().loc[params.index]

    lower = params - conf[0]
    upper = conf[1] - params

    y_pos = np.arange(len(params))

    plt.figure(figsize=(8, 10))

    plt.errorbar(params, y_pos, xerr=[lower, upper], fmt='o')
    plt.axvline(0)

    plt.yticks(y_pos, params.index)
    plt.xlabel("Effect on log(cost)")
    plt.title(title)

    plt.tight_layout()

    path = f"{SAVE_DIR}/{filename}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")

    print("Saved:", path)
    plt.close()


# ======================================================
# 2) DISTRIBUTION PLOT (SAVED)
# ======================================================

def plot_distribution(series, title, filename):

    plt.figure(figsize=(7,5))

    plt.hist(series.dropna(), bins=50)

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(title)

    plt.tight_layout()

    path = f"{SAVE_DIR}/{filename}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")

    print("Saved:", path)
    plt.close()


# ======================================================
# 3) GENERATE ALL PLOTS
# ======================================================

# coefficient plots
plot_coefficients(model_short_log, "Short Illness → Log Cost Effects", "coef_short_log")
plot_coefficients(model_major_log, "Major Illness → Log Cost Effects", "coef_major_log")

# distributions
plot_distribution(df["short_total_cost"], "Short Cost Distribution (Raw)", "short_raw_dist")
plot_distribution(df["log_short_cost"], "Short Cost Distribution (Log)", "short_log_dist")

plot_distribution(df["major_total_cost"], "Major Cost Distribution (Raw)", "major_raw_dist")
plot_distribution(df["log_major_cost"], "Major Cost Distribution (Log)", "major_log_dist")
