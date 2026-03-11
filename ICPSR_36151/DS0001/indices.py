import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ================================
# LOAD DATA
# ================================
data_path = "states_datasets/data_subset_Uttar Pradesh 09.csv"
df = pd.read_csv(data_path, low_memory=False)

# ================================
# 1) IDENTIFY VARIABLES
# ================================

# Binary health conditions
binary_health = [
    "SM4","SM5","SM6","SM7","SM8","SM9","SM10","SM12",
    "MB3","MB4","MB5","MB6","MB7","MB8","MB9","MB10",
    "MB11","MB12","MB13","MB14","MB15","MB16","MB17"
]

# Days/severity variables
severity = [
    "SM3",   # days ill
    "SM11",  # days disabled last 30 days
    "MB18",  # days disabled last 12 months
    "MB24"   # days hospitalized (major)
]

# Financial cost variables
cost_vars = [
    "SM18","SM20","SM21","SM22",
    "MB25","MB26","MB27","MB28","MB29"
]

# ================================
# 2) CLEAN & RECODE
# ================================

# Recode binary presence (1=yes, 2=no)
def recode_binary(x):
    return x.replace({1:1,2:0}).fillna(0)

for col in binary_health:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = recode_binary(df[col])
    else:
        print(f"WARNING: {col} not found")

# Numeric severity & cost
for col in severity + cost_vars:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f"WARNING: {col} not found")

# ================================
# 3) BUILD SUB-INDICES
# ================================

# 3a) Binary health burden = percent of conditions present
df["binary_burden"] = df[binary_health].sum(axis=1) / len(binary_health)

# 3b) Severity = normalized average of days ill/disability/hospital
df["severity_sum"] = df[severity].fillna(0).sum(axis=1)
# normalize to 0–1
df["severity_norm"] = MinMaxScaler().fit_transform(df[["severity_sum"]])

# 3c) Financial burden (log scale reduces skew)
df["cost_sum"] = df[cost_vars].fillna(0).sum(axis=1)
df["cost_log"] = np.log1p(df["cost_sum"])  # log(1 + cost)
df["cost_norm"] = MinMaxScaler().fit_transform(df[["cost_log"]])

# ================================
# 4) COMPOSITE HEALTH INDEX
# ================================
# weights
w1, w2, w3 = 0.5, 0.3, 0.2

df["health_index"] = (
    1 
    - (w1 * df["binary_burden"]
       + w2 * df["severity_norm"]
       + w3 * df["cost_norm"])
)
df["health_index"] = df["health_index"].clip(0,1)

# ================================
# 5) CORRELATE WITH AGE
# ================================
df["RO5"] = pd.to_numeric(df["RO5"], errors="coerce")
age_corr = df["RO5"].corr(df["health_index"])
print(f"Correlation with AGE: {age_corr:.4f}")

# ================================
# 6) SAVE OUTPUT
# ================================
cols_out = [
    "RO5",
    "binary_burden","severity_norm","cost_norm","health_index"
]
df[cols_out].to_csv("health_index_refined.csv", index=False)

print("Saved health_index_refined.csv")

# ================================
# 7) PLOT
# ================================
plt.hist(df["health_index"].dropna(), bins=50)
plt.title("Refined Health Index Distribution")
plt.xlabel("Health Index (0 = worst, 1 = best)")
plt.ylabel("Frequency")
plt.savefig("health_index_refined_plot.png")
plt.show()
