import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm


# ======================================================
# 1) LOAD DATA
# ======================================================
data_path = "states_datasets/data_subset_Uttar Pradesh 09.csv"

df = pd.read_csv(data_path, low_memory=False)

print("Loaded shape:", df.shape)


# ======================================================
# 2) VARIABLES
# ======================================================
'''
fever: SM4
Cough: SM5
Cough with short breath: SM6
Diarrhea: SM7
Bloody Diarrhea: SM8
Cataract: MB3
TB: MB4
BP: MB5
Heart Disease: MB6
Diabetes: MB7
Leprosy: MB8
Cancer:MB9
Asthma: MB10
Polio: MB11
Paralysis: MB12
Epilepsy: MB13 
Mental illness: MB14 
STD or AIDS: MB15 
Accident: MB16
Other Major: MB17
'''
binary_health = [
    "SM4","SM5","SM6","SM7","SM8","MB3","MB4",
    "MB5","MB6","MB7","MB8","MB9","MB10","MB11",
    "MB12","MB13","MB14","MB15","MB16","MB17"
]

cost_vars = [
    "SM18","SmM19","SM20","SM21","SM22",
    "MB25","MB26","MB27","MB28","MB29"
]
Major_binary_health = [
    "MB3","MB4",
    "MB5","MB6","MB7","MB8","MB9","MB10","MB11",
    "MB12","MB13","MB14","MB15","MB16","MB17"
]
Short_Binary_health = [
    "SM4","SM5","SM6","SM7","SM8"
]
short_cost_var = [
    "SM18","SmM19","SM20","SM21","SM22",
]
Major_cost_vars = [
    "MB25","MB26","MB27","MB28","MB29"
]

severity_vars = ["SM11", "MB18", "MB24"]

age_var = "RO5"
gender_var = "RO3"
education_vars = ["ED2","ED3","ED4"]

# ======================================================
# 3) NUMERIC CONVERSION (EXCEPT GENDER!!)
# ======================================================

def safe_binary(cols):
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.extract(r'(\d)$')[0]   # take last digit (0/1)
                .astype(float)
            )

safe_binary(binary_health)

def safe_numeric(cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

safe_numeric(severity_vars + cost_vars + [age_var] + education_vars)


# ======================================================
# 4) BINARY RECODE
# ======================================================

def recode_binary(s):
    return s.replace({1:1, 2:0}).fillna(0)

for col in binary_health:
    if col in df.columns:
        df[col] = recode_binary(df[col])


# ======================================================
# 5) GENDER (FOR "Male 1" / "Female 2")
# ======================================================

print("\nRaw gender values:")
print(df[gender_var].value_counts(dropna=False))

# Extract first word only
df["gender"] = df[gender_var].astype(str).str.split().str[0]

print("\nClean gender counts:")
print(df["gender"].value_counts(dropna=False))


# ======================================================
# 6) BUILD SUB-INDICES
# ======================================================

df["binary_burden"] = df[binary_health].sum(axis=1)

df["severity_sum"] = df[severity_vars].fillna(0).sum(axis=1)
df["cost_sum"] = df[cost_vars].fillna(0).sum(axis=1)

scaler = MinMaxScaler()

df["severity_norm"] = scaler.fit_transform(df[["severity_sum"]])

df["cost_log"] = np.log1p(df["cost_sum"])
df["cost_norm"] = scaler.fit_transform(df[["cost_log"]])


# ======================================================
# 7) PCA HEALTH INDEX
# ======================================================

features = df[["binary_burden","severity_norm","cost_norm"]].fillna(0)

features_std = StandardScaler().fit_transform(features)

pca = PCA(n_components=1)

df["health_pca"] = pca.fit_transform(features_std)
df["health_pca_norm"] = scaler.fit_transform(df[["health_pca"]])

print("\nPCA weights:", pca.components_)


# ======================================================
# 8) REGRESSION ADJUSTMENT
# ======================================================

# ======================================================
# 8) REGRESSION ADJUSTMENT  (SAFE VERSION)
# ======================================================

df[age_var] = df[age_var].fillna(df[age_var].median())

# Base numeric predictors
X = df[[age_var] + education_vars].fillna(0)

# Gender dummy
gender_dummies = pd.get_dummies(df["gender"], drop_first=True)

X = pd.concat([X, gender_dummies], axis=1)

# 🔥 CRITICAL FIX
X = X.astype(float)

# add intercept
X = sm.add_constant(X)

y = df["health_pca_norm"].astype(float)

reg = sm.OLS(y, X).fit()

print(reg.summary())


df["health_adjusted"] = reg.predict(X)
df["health_adjusted_norm"] = scaler.fit_transform(df[["health_adjusted"]])


# ======================================================
# FIX FRAGMENTATION WARNING
# ======================================================
df = df.copy()


# ======================================================
# 9) GROUP COMPARISONS
# ======================================================

# Age groups
df["age_group"] = pd.cut(
    df[age_var],
    bins=[0,18,35,60,100],
    labels=["0-17","18-34","35-59","60+"]
)

age_means = (
    df.groupby("age_group", observed=True)["health_adjusted_norm"]
    .mean()
    .reset_index()
)


# Gender means (NOW ALWAYS WORKS)
gender_means = (
    df.groupby("gender")["health_adjusted_norm"]
    .mean()
    .reset_index()
)


# Education
edu_means = {}
for ed in education_vars:
    if ed in df.columns:
        edu_means[ed] = (
            df.groupby(ed)["health_adjusted_norm"]
            .mean()
            .reset_index()
        )


# Quantiles
df["health_quantile"] = pd.qcut(df["health_adjusted_norm"], 5, labels=False) + 1


# ======================================================
# 10) SAVE CSVs
# ======================================================

cols_to_save = [
    age_var, gender_var,
    "binary_burden",
    "severity_norm",
    "cost_norm",
    "health_pca_norm",
    "health_adjusted_norm",
    "health_quantile"
]

df[cols_to_save].to_csv("health_full_analysis_output.csv", index=False)
age_means.to_csv("age_vs_health.csv", index=False)
gender_means.to_csv("gender_vs_health.csv", index=False)

for ed, dfe in edu_means.items():
    dfe.to_csv(f"edu_{ed}_vs_health.csv", index=False)

print("\nSaved all CSV files ✓")


# ======================================================
# 11) PLOTS
# ======================================================

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
sns.barplot(data=age_means, x="age_group", y="health_adjusted_norm")
plt.title("Health Index by Age Group")

plt.subplot(2,2,2)
sns.barplot(data=gender_means, x="gender", y="health_adjusted_norm")
plt.title("Health Index by Gender")

plt.subplot(2,2,3)
sns.histplot(df["health_quantile"], bins=5)
plt.title("Health Quantile Distribution")

plt.tight_layout()
plt.savefig("health_full_analysis_plots.png", dpi=300)
plt.show()

print("\nALL DONE ✓")
