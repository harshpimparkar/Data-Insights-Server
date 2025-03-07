from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# Initialize Flask App
app = Flask(__name__)

class DataInsightsGenerator:
    def __init__(self, csv_path):
        """Initialize with path to CSV file."""
        self.df = pd.read_csv(csv_path)
        self.insights = {}
        
    def generate_descriptive_analytics(self):
        """Generate descriptive statistics and data quality metrics."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        descriptive_stats = {
            'basic_stats': self.df[numeric_cols].describe().to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'unique_values': {col: self.df[col].nunique() for col in self.df.columns},
            'data_types': self.df.dtypes.astype(str).to_dict()
        }
        
        for col in numeric_cols:
            descriptive_stats[f'{col}_skewness'] = stats.skew(self.df[col].dropna())
            descriptive_stats[f'{col}_kurtosis'] = stats.kurtosis(self.df[col].dropna())
        
        self.insights['descriptive'] = descriptive_stats
        return descriptive_stats

    def generate_diagnostic_analytics(self):
        """Perform correlation analysis and identify potential relationships."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        diagnostic_insights = {
            'correlation_matrix': self.df[numeric_cols].corr().to_dict(),
            'strong_correlations': []
        }
        
        corr_matrix = self.df[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    diagnostic_insights['strong_correlations'].append({
                        'variables': (numeric_cols[i], numeric_cols[j]),
                        'correlation': corr_value
                    })
        
        self.insights['diagnostic'] = diagnostic_insights
        return diagnostic_insights

    def generate_predictive_analytics(self, target_column, forecast_periods=30):
        """Generate predictions using various models."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        features = [col for col in numeric_cols if col != target_column]
        
        predictive_insights = {}
        
        if len(features) > 0:
            X = self.df[features]
            y = self.df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            predictive_insights['feature_importance'] = dict(zip(features, model.coef_))
            predictive_insights['model_accuracy'] = {
                'r2_score': r2_score(y_test, model.predict(X_test)),
                'rmse': np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
            }
        
        date_cols = self.df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_col = date_cols[0]
            self.df.set_index(date_col, inplace=True)
            
            model = ExponentialSmoothing(
                self.df[target_column],
                seasonal_periods=12,
                trend='add',
                seasonal='add'
            ).fit()
            
            forecast = model.forecast(forecast_periods)
            predictive_insights['time_series_forecast'] = forecast.to_dict()
        
        self.insights['predictive'] = predictive_insights
        return predictive_insights

    def generate_prescriptive_analytics(self):
        """Generate actionable recommendations based on insights."""
        prescriptive_insights = {
            'recommendations': [],
            'optimization_opportunities': [],
            'risk_factors': []
        }
        
        if 'predictive' in self.insights:
            feature_importance = self.insights['predictive'].get('feature_importance', {})
            
            for feature, importance in feature_importance.items():
                if abs(importance) > 0.5:
                    prescriptive_insights['recommendations'].append({
                        'feature': feature,
                        'importance': importance,
                        'action': f"Focus on optimizing {feature} due to high impact"
                    })
        
        if 'diagnostic' in self.insights:
            strong_correlations = self.insights['diagnostic'].get('strong_correlations', [])
            
            for corr in strong_correlations:
                prescriptive_insights['optimization_opportunities'].append({
                    'variables': corr['variables'],
                    'correlation': corr['correlation'],
                    'suggestion': f"Consider joint optimization of {corr['variables'][0]} and {corr['variables'][1]}"
                })
        
        self.insights['prescriptive'] = prescriptive_insights
        return prescriptive_insights

    def detect_outliers(self):
        """
        Detect outliers using Z-score and IQR methods.
        Returns serializable dictionary instead of DataFrames.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outliers = {
            'z_score_outliers': {
                'count': 0,
                'indices': [],
                'values': {}
            },
            'iqr_outliers': {
                'count': 0,
                'indices': [],
                'values': {}
            }
        }
        
        # Z-score Method
        z_scores = np.abs(stats.zscore(self.df[numeric_cols]))
        z_outliers = (z_scores > 3).any(axis=1)
        z_outlier_df = self.df[z_outliers]
        
        # Convert Z-score outliers to serializable format
        if not z_outlier_df.empty:
            outliers['z_score_outliers']['count'] = len(z_outlier_df)
            outliers['z_score_outliers']['indices'] = z_outlier_df.index.tolist()
            outliers['z_score_outliers']['values'] = z_outlier_df.to_dict(orient='records')
        
        # IQR Method
        Q1 = self.df[numeric_cols].quantile(0.25)
        Q3 = self.df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = ((self.df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                       (self.df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        iqr_outlier_df = self.df[iqr_outliers]
        
        # Convert IQR outliers to serializable format
        if not iqr_outlier_df.empty:
            outliers['iqr_outliers']['count'] = len(iqr_outlier_df)
            outliers['iqr_outliers']['indices'] = iqr_outlier_df.index.tolist()
            outliers['iqr_outliers']['values'] = iqr_outlier_df.to_dict(orient='records')
        
        # Add summary statistics
        outliers['summary'] = {
            'total_rows': len(self.df),
            'z_score_outlier_percentage': (len(z_outlier_df) / len(self.df) * 100) if len(self.df) > 0 else 0,
            'iqr_outlier_percentage': (len(iqr_outlier_df) / len(self.df) * 100) if len(self.df) > 0 else 0,
            'columns_analyzed': numeric_cols.tolist()
        }
        
        return outliers

    def to_json_serializable(self):
        """
        Convert all insights to JSON serializable format.
        """
        serializable_insights = {}
        
        # Handle numeric values that might not be JSON serializable
        for insight_type, data in self.insights.items():
            if isinstance(data, dict):
                serializable_insights[insight_type] = self._convert_to_serializable(data)
        
        return serializable_insights
    
    def _convert_to_serializable(self, obj):
        """
        Recursively convert objects to JSON serializable format.
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
