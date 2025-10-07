# src/utils/data_processor.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class DataProcessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.original_shape = df.shape
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing': self.df.isnull().sum().to_dict(),
            'memory': self.df.memory_usage(deep=True).sum()
        }
    
    def handle_missing_values(self, method: str = 'drop') -> pd.DataFrame:
        """Handle missing values"""
        if method == 'drop':
            self.df = self.df.dropna()
        elif method == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(
                self.df[numeric_cols].mean()
            )
        elif method == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(
                self.df[numeric_cols].median()
            )
        return self.df
    
    def remove_outliers(self, columns: list = None) -> pd.DataFrame:
        """Remove outliers using IQR method"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.df = self.df[
                (self.df[col] >= lower_bound) & 
                (self.df[col] <= upper_bound)
            ]
        
        return self.df
    
    def normalize_data(self, method: str = 'minmax') -> pd.DataFrame:
        """Normalize numeric columns"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            self.df[numeric_cols] = (
                (self.df[numeric_cols] - self.df[numeric_cols].min()) /
                (self.df[numeric_cols].max() - self.df[numeric_cols].min())
            )
        elif method == 'standard':
            self.df[numeric_cols] = (
                (self.df[numeric_cols] - self.df[numeric_cols].mean()) /
                self.df[numeric_cols].std()
            )
        
        return self.df
    
    def encode_categorical(self, method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical variables"""
        if method == 'onehot':
            self.df = pd.get_dummies(self.df, drop_first=True)
        return self.df
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix for numeric columns"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.corr()