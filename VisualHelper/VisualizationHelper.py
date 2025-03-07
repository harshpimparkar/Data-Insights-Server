import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import matplotlib
# Set the backend to 'Agg' before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import os
from datetime import datetime
from threading import Lock

# Create a lock for thread-safe plotting
plot_lock = Lock()

def analyze_dataframe_structure(df: pd.DataFrame) -> Dict:
    """Analyze DataFrame structure for visualization suggestions"""
    column_types = {
        'numeric': list(df.select_dtypes(include=['int64', 'float64']).columns),
        'categorical': list(df.select_dtypes(include=['object', 'category']).columns),
        'datetime': list(df.select_dtypes(include=['datetime64']).columns),
        'boolean': list(df.select_dtypes(include=['bool']).columns)
    }
    
    return {
        'column_types': column_types,
        'row_count': len(df),
        'column_count': len(df.columns)
    }

def create_visualization(df: pd.DataFrame, viz_type: str, columns: Dict, title: str) -> Dict:
    """Create and save visualization based on type"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = 'static/visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with plot_lock:  # Use lock for thread-safe plotting
            if viz_type == 'correlation':
                fig, ax = plt.subplots(figsize=(10, 8))
                corr = df[columns['numeric']].corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax)
                ax.set_title(title)
                filename = f'{output_dir}/correlation_{timestamp}.png'
                fig.savefig(filename)
                plt.close(fig)
                
            elif viz_type == 'distribution':
                fig, ax = plt.subplots(figsize=(12, 6))
                for col in columns['numeric'][:3]:  # Limit to 3 columns
                    sns.kdeplot(data=df[col], label=col, ax=ax)
                ax.set_title(title)
                ax.legend()
                filename = f'{output_dir}/distribution_{timestamp}.png'
                fig.savefig(filename)
                plt.close(fig)
                
            elif viz_type == 'categorical':
                fig, ax = plt.subplots(figsize=(12, 6))
                cat_col = columns['categorical'][0]
                num_col = columns['numeric'][0]
                sns.boxplot(data=df, x=cat_col, y=num_col, ax=ax)
                ax.set_title(title)
                plt.xticks(rotation=45)
                filename = f'{output_dir}/categorical_{timestamp}.png'
                fig.savefig(filename)
                plt.close(fig)
                
            elif viz_type == 'time_series':
                # Plotly is thread-safe, no need for lock
                fig = px.line(df, x=columns['datetime'][0], y=columns['numeric'][0],
                             title=title)
                filename = f'{output_dir}/timeseries_{timestamp}.png'
                pio.write_image(fig, filename)
                
            elif viz_type == 'pie_chart':
                cat_col = columns['categorical'][0]
                num_col = columns['numeric'][0]
                pie_data = df.groupby(cat_col)[num_col].sum().reset_index()
                fig = px.pie(pie_data, names=cat_col, values=num_col, title=title)
                filename = f'{output_dir}/piechart_{timestamp}.png'
                pio.write_image(fig, filename)
            
            elif viz_type == 'bar_chart':
                cat_col = columns['categorical'][0]
                num_col = columns['numeric'][0]
                bar_data = df.groupby(cat_col)[num_col].sum().reset_index()
                fig = px.bar(bar_data, x=cat_col, y=num_col, title=title)
                filename = f'{output_dir}/barchart_{timestamp}.png'
                pio.write_image(fig, filename)

        return {
            'status': 'success',
            'filename': filename,
            'type': viz_type,
            'title': title
        }
        
    except Exception as e:
        logging.error(f"Error creating visualization: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'type': viz_type
        }

def generate_visualizations(df: pd.DataFrame, insights: Dict) -> List[Dict]:
    """Generate and save all relevant visualizations"""
    df_analysis = analyze_dataframe_structure(df)
    visualizations = []
    
    # Correlation heatmap for numeric columns
    if len(df_analysis['column_types']['numeric']) >= 2:
        viz = create_visualization(
            df,
            'correlation',
            {'numeric': df_analysis['column_types']['numeric']},
            'Correlation Analysis'
        )
        if viz['status'] == 'success':
            visualizations.append(viz)
    
    # Distribution plots for numeric columns
    if df_analysis['column_types']['numeric']:
        viz = create_visualization(
            df,
            'distribution',
            {'numeric': df_analysis['column_types']['numeric']},
            'Distribution Analysis'
        )
        if viz['status'] == 'success':
            visualizations.append(viz)
    
    # Categorical analysis
    if df_analysis['column_types']['categorical'] and df_analysis['column_types']['numeric']:
        viz = create_visualization(
            df,
            'categorical',
            {
                'categorical': df_analysis['column_types']['categorical'],
                'numeric': df_analysis['column_types']['numeric']
            },
            'Categorical Analysis'
        )
        if viz['status'] == 'success':
            visualizations.append(viz)
    
    # Time series if datetime column exists
    if df_analysis['column_types']['datetime'] and df_analysis['column_types']['numeric']:
        viz = create_visualization(
            df,
            'time_series',
            {
                'datetime': df_analysis['column_types']['datetime'],
                'numeric': df_analysis['column_types']['numeric']
            },
            'Time Series Analysis'
        )
        if viz['status'] == 'success':
            visualizations.append(viz)
    
    return visualizations