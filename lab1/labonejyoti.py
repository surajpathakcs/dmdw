# -*- coding: utf-8 -*-
"""
Lab 1: Exploratory Data Analysis (EDA)
Datasets: Iris Dataset and Haberman Dataset
Author: Jyoti
"""

# -------------------------------
# 1. Import Libraries
# -------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# Lab completed by Jyoti
# -------------------------------
print("\nLab completed by: Jyoti\n")

# -------------------------------
# 2. Iris Dataset Overview
# -------------------------------
iris = pd.read_csv("iris.csv")  # Local CSV file

print("Iris Dataset Shape:", iris.shape)
print("Columns:", iris.columns)

print("\nClass Distribution:\n", iris["species"].value_counts())
print("\nBasic Statistics:\n", iris.describe())

# -------------------------------
# 3. Univariate Analysis
# -------------------------------
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# 3.1 Histograms by species
for feature in features:
    sns.FacetGrid(iris, hue="species", height=5)\
       .map(sns.histplot, feature, kde=True)\
       .add_legend()
    plt.title(f"Histogram of {feature} by Species (Jyoti)", y=0.95)
    plt.tight_layout()
    plt.show()
    plt.close()

# 3.2 PDF and CDF for petal_length
species_list = iris['species'].unique()
for sp in species_list:
    data = iris.loc[iris['species'] == sp]['petal_length']
    counts, bin_edges = np.histogram(data, bins=10, density=True)
    pdf = counts / sum(counts)
    cdf = np.cumsum(pdf)
    plt.plot(bin_edges[1:], pdf, label="PDF")
    plt.plot(bin_edges[1:], cdf, label="CDF")
    plt.title(f"PDF and CDF of petal_length for {sp} (Jyoti)", y=0.95)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

# 3.3 Mean, Standard Deviation
print("\nMeans:")
for sp in species_list:
    mean_val = np.mean(iris.loc[iris['species']==sp]['petal_length'])
    print(f"{sp}: {mean_val:.2f}")

print("\nStandard Deviations:")
for sp in species_list:
    std_val = np.std(iris.loc[iris['species']==sp]['petal_length'])
    print(f"{sp}: {std_val:.2f}")

# 3.4 Median, Percentiles, IQR
print("\nMedians and Percentiles:")
for sp in species_list:
    data = iris.loc[iris['species']==sp]['petal_length']
    median_val = np.median(data)
    percentile_25 = np.percentile(data, 25)
    percentile_75 = np.percentile(data, 75)
    iqr = percentile_75 - percentile_25
    print(f"{sp}: Median={median_val}, IQR={iqr}, 25th={percentile_25}, 75th={percentile_75}")

# 3.5 Boxplot
sns.boxplot(x='species', y='petal_length', data=iris)
plt.title("Boxplot of Petal Length by Species (Jyoti)", y=0.95)
plt.tight_layout()
plt.show()
plt.close()

# 3.6 Violin Plot
sns.violinplot(x='species', y='petal_length', data=iris)
plt.title("Violin Plot of Petal Length by Species (Jyoti)", y=0.95)
plt.tight_layout()
plt.show()
plt.close()

# -------------------------------
# 4. Bivariate Analysis
# -------------------------------
# 4.1 2-D Scatter Plots (all feature combinations)
for i, feature_x in enumerate(features):
    for feature_y in features[i+1:]:
        sns.scatterplot(data=iris, x=feature_x, y=feature_y, hue='species')
        plt.title(f"Scatter plot of {feature_x} vs {feature_y} (Jyoti)", y=0.95)
        plt.tight_layout()
        plt.show()
        plt.close()

# 4.2 Pair-Plot
sns.pairplot(iris, hue='species', height=2.5)
plt.suptitle("Pairplot of Iris Features (Jyoti)", y=0.95)
plt.tight_layout()
plt.show()
plt.close()

# -------------------------------
# 5. Haberman Dataset Overview
# -------------------------------
haberman = pd.read_csv("haberman.csv")  # Local CSV file

print("Haberman Dataset Shape:", haberman.shape)
print("Columns:", haberman.columns)

# Map numeric status to categorical
haberman['status'] = haberman['status'].replace([1, 2], ['yes', 'no'])

# Class distribution
print("\nClass Distribution:\n", haberman['status'].value_counts())

# Basic statistics
print("\nBasic Statistics:\n", haberman.describe())
