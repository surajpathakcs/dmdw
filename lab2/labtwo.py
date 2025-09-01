# lab2_preprocessing_v1.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

print("\n================== LAB 2 PREPROCESSING ==================\n")

# 1. Binning
print("--- 1. BINNING ---")
df_ages = pd.DataFrame({'age': np.random.randint(12, 85, 12)})
print("Original Ages:\n", df_ages.to_string(index=False))

df_ages['age_bins'] = pd.cut(df_ages['age'], bins=8)
print("\nEqual-width Bins:\n", df_ages[['age','age_bins']].to_string(index=False))

df_ages['custom_bins'] = pd.cut(df_ages['age'], bins=[15, 25, 35, 45, 60, 85])
print("\nCustom Bins:\n", df_ages[['age','custom_bins']].to_string(index=False))

df_ages['life_stage'] = pd.cut(
    df_ages['age'], bins=[0, 12, 19, 35, 55, 100],
    labels=['child','teenager','young_adult','adult','senior']
)
print("\nCategorical Life Stages:\n", df_ages[['age','life_stage']].to_string(index=False))
print("\nLife Stage Counts:\n", df_ages['life_stage'].value_counts())

# 2. Missing Value Imputation
print("\n--- 2. MISSING VALUE IMPUTATION ---")
X = np.array([6, np.nan, 8, 2, np.nan, 10])
print("Original Array:", X)

mean_imp = SimpleImputer(strategy='mean')
print("Mean Imputation:", mean_imp.fit_transform(X.reshape(-1,1)).ravel())

median_imp = SimpleImputer(strategy='median')
print("Median Imputation:", median_imp.fit_transform(X.reshape(-1,1)).ravel())

X2 = np.array([[2, np.nan],[3, 6],[np.nan, 8]])
mode_imp = SimpleImputer(strategy='most_frequent')
print("Mode Imputation:\n", mode_imp.fit_transform(X2))

# 3. Standardization
print("\n--- 3. STANDARDIZATION ---")
X_train = np.array([[3., -2., 6.],[4., 1., 1.],[2., 3., -1.]])
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
print("Original Data:\n", X_train)
print("\nStandardized Data:\n", X_scaled)
print("Means:", X_scaled.mean(axis=0))
print("Std Devs:", X_scaled.std(axis=0))

# 4. Min-Max Normalization
print("\n--- 4. MIN-MAX NORMALIZATION ---")
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X_train)
print("Min-Max Normalized Data:\n", X_minmax)

# 5. One Hot Encoding
print("\n--- 5. ONE HOT ENCODING ---")
df = pd.read_csv("weather_data.csv")
print("Original Weather Data:\n", df.head())

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['event']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['event']))
df_final = pd.concat([df, encoded_df], axis=1)
print("\nAfter One-Hot Encoding:\n", df_final.head())
