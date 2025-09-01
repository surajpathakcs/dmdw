# lab2_preprocessing_v2.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

print("\n########## LAB 2: DATA PREPROCESSING ##########\n")

# 1. Binning
print(">>> 1. BINNING <<<")
df_ages = pd.DataFrame({'age': np.random.randint(10, 90, 12)})
print("Original Age Data:\n", df_ages.to_string(index=False))

df_ages['age_bins'] = pd.cut(df_ages['age'], bins=7)
print("\nEqual-width Binning:\n", df_ages[['age','age_bins']].to_string(index=False))

df_ages['custom_bins'] = pd.cut(df_ages['age'], bins=[10, 20, 30, 40, 50, 70, 90])
print("\nCustom Bins:\n", df_ages[['age','custom_bins']].to_string(index=False))

df_ages['life_stage'] = pd.cut(
    df_ages['age'], bins=[0, 13, 20, 36, 60, 100],
    labels=['child','teen','young_adult','adult','senior']
)
print("\nLife Stage Categories:\n", df_ages[['age','life_stage']].to_string(index=False))
print("\nLife Stage Distribution:\n", df_ages['life_stage'].value_counts())

# 2. Missing Value Imputation
print("\n>>> 2. MISSING VALUE IMPUTATION <<<")
X = np.array([7, np.nan, 5, np.nan, 11, 3])
print("Original Array:", X)

mean_imp = SimpleImputer(strategy='mean')
print("Mean Imputation:", mean_imp.fit_transform(X.reshape(-1,1)).ravel())

median_imp = SimpleImputer(strategy='median')
print("Median Imputation:", median_imp.fit_transform(X.reshape(-1,1)).ravel())

X2 = np.array([[1, np.nan],[4, 5],[np.nan, 9]])
mode_imp = SimpleImputer(strategy='most_frequent')
print("Mode Imputation:\n", mode_imp.fit_transform(X2))

# 3. Standardization
print("\n>>> 3. STANDARDIZATION <<<")
X_train = np.array([[5., -3., 7.],[3., 0., 2.],[1., 2., -1.]])
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = np.round(scaler.transform(X_train), 3)
print("Original Data:\n", X_train)
print("\nStandardized Data:\n", X_scaled)
print("Column Means:", np.round(X_scaled.mean(axis=0), 3))
print("Column Stds:", np.round(X_scaled.std(axis=0), 3))

# 4. Min-Max Normalization
print("\n>>> 4. MIN-MAX NORMALIZATION <<<")
X_minmax = preprocessing.MinMaxScaler().fit_transform(X_train)
print("Min-Max Normalized Data:\n", np.round(X_minmax,3))

# 5. One Hot Encoding
print("\n>>> 5. ONE HOT ENCODING <<<")
df = pd.read_csv("weather_data.csv")
print("Weather Data Sample:\n", df.head())

encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[['event']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['event']))
df_final = pd.concat([df, encoded_df], axis=1)
print("\nAfter One-Hot Encoding:\n", df_final.head())
