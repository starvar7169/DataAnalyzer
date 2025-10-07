"""File upload and reading utilities"""

import pandas as pd
import streamlit as st
from typing import Optional

class FileHandler:
    """Handles file upload and reading"""
    
    @staticmethod
    def read_file(uploaded_file) -> Optional[pd.DataFrame]:
        """Read uploaded file and return DataFrame"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension == 'xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension == 'txt':
                df = pd.read_csv(uploaded_file, delimiter='\t')
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return None
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> bool:
        """Validate DataFrame is usable"""
        if df is None or df.empty:
            return False
        if len(df.columns) == 0:
            return False
        return True
    