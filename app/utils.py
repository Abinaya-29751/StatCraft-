# Optional: Add helper functions here if needed
import pandas as pd
import numpy as np

def detect_outliers(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def calculate_missing_percentage(df):
    """Calculate percentage of missing values per column"""
    missing_pct = (df.isnull().sum() / len(df)) * 100
    return missing_pct.to_dict()
