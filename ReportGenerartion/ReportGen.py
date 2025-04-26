import os
import logging
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List
import google.generativeai as genai
from xhtml2pdf import pisa
import markdown
import tempfile

class InsightReportGenerator:
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "reports"):
        """Initialize with visualization output directory."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not provided")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.configure_gemini()
        self.logger = logging.getLogger(__name__)
        plt.switch_backend('Agg')  # Non-interactive backend

    def configure_gemini(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def generate_full_report(self, df: pd.DataFrame, insights: Dict[str, Any], 
                           dataset_name: str, target_column: Optional[str] = None,
                           output_format: str = "pdf") -> Dict[str, Any]:
        """Generate complete report with visualizations."""
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir = os.path.join(self.output_dir, report_id)
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate visualizations
        viz_data = self.generate_visualizations(df, insights, report_dir)
        
        # Generate report content in markdown
        report_content = self._generate_report_content(
            insights=insights,
            visualization_paths=viz_data["paths"],
            dataset_name=dataset_name,
            target_column=target_column
        )
        
        # Save markdown report to file
        report_md_path = os.path.join(report_dir, "report.md")
        with open(report_md_path, 'w') as f:
            f.write(report_content)
        
        result = {
            "report_content": report_content,
            "report_markdown_path": report_md_path,
            "visualization_paths": viz_data["paths"],
            "report_id": report_id
        }
        
        # Generate PDF if requested
        if output_format == "pdf":
            pdf_path = self.convert_markdown_to_pdf(report_content, viz_data["paths"], report_dir)
            result["report_pdf_path"] = pdf_path
        
        return result

    def convert_markdown_to_pdf(self, markdown_content: str, viz_paths: Dict[str, str], 
                      output_dir: str) -> str:
        """Convert markdown report to PDF with enhanced user-friendly styling."""
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content, extensions=['toc', 'tables'])
        
        # Fix image references in HTML
        for viz_name, viz_path in viz_paths.items():
            # Convert to proper base64 data to embed images directly
            try:
                with open(viz_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                    image_type = os.path.splitext(viz_path)[1].lstrip('.')
                    if image_type.lower() == 'png':
                        image_data = f'data:image/png;base64,{encoded_image}'
                    elif image_type.lower() in ['jpg', 'jpeg']:
                        image_data = f'data:image/jpeg;base64,{encoded_image}'
                    else:
                        image_data = f'data:image/{image_type};base64,{encoded_image}'
                    
                html_content = html_content.replace(f'![]({viz_path})', 
                    f'<div class="figure-container"><img src="{image_data}" alt="{viz_name}" class="centered-image"><div class="figure-caption">{viz_name}</div></div>')
            except Exception as e:
                self.logger.error(f"Failed to embed image {viz_path}: {str(e)}")
        
        # Replace TOC placeholder if exists
        if '[TOC]' in html_content:
            # Create simple TOC since xhtml2pdf doesn't support the extension's TOC
            toc_entries = []
            import re
            headers = re.findall(r'<h([1-3])[^>]*>(.*?)</h\1>', html_content)
            for level, title in headers:
                anchor = title.lower().replace(' ', '-').replace('.', '').replace(',', '')
                toc_entries.append(f'<li class="toc-h{level}"><a href="#{anchor}">{title}</a></li>')
                # Add anchors to headers
                html_content = html_content.replace(f'<h{level}>{title}</h{level}>', 
                                                f'<h{level} id="{anchor}">{title}</h{level}>')
            
            toc_html = f'<div class="toc"><h2>Table of Contents</h2><ul>{"".join(toc_entries)}</ul></div>'
            html_content = html_content.replace('[TOC]', toc_html)
        
        # Add enhanced styling for user-friendly presentation
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    line-height: 1.4; 
                    margin: 40px; 
                    font-size: 11pt;
                    color: #333;
                    background-color: #ffffff;
                }}
                h1 {{ 
                    color: #2c3e50; 
                    margin-bottom: 15px; 
                    margin-top: 25px;
                    font-size: 24pt;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 5px;
                }}
                h2 {{ 
                    color: #2c3e50; 
                    border-bottom: 1px solid #bdc3c7; 
                    padding-bottom: 5px; 
                    margin-top: 20px;
                    margin-bottom: 15px;
                    font-size: 18pt;
                }}
                h3 {{ 
                    color: #34495e; 
                    margin-top: 15px;
                    margin-bottom: 10px;
                    font-size: 14pt;
                }}
                p {{ 
                    margin: 10px 0; 
                    text-align: justify;
                }}
                .highlight {{
                    background-color: #f9f9f9;
                    border-left: 3px solid #3498db;
                    padding: 10px;
                    margin: 15px 0;
                }}
                .figure-container {{
                    margin: 20px auto;
                    text-align: center;
                    width: 80%;  /* MODIFIED: Reduced from 100% to 80% */
                    display: block;
                }}
                .centered-image {{
                    max-width: 80%;  /* MODIFIED: Reduced from 100% to 80% */
                    height: auto;
                    margin: 0 auto;
                    display: block;
                }}
                .figure-caption {{
                    font-style: italic;
                    color: #7f8c8d;
                    font-size: 10pt;
                    margin-top: 5px;
                    text-align: center;  /* Added to center the caption */
                }}
                img {{ 
                    max-width: 80%;  /* MODIFIED: Reduced from 100% to 80% */
                    height: auto; 
                    display: block;
                    margin: 0 auto;
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 15px 0; 
                    font-size: 10pt;
                }}
                th, td {{ 
                    border: 1px solid #dddddd; 
                    text-align: left; 
                    padding: 8px; 
                }}
                th {{ 
                    background-color: #f2f2f2; 
                    color: #2c3e50;
                }}
                ul, ol {{ 
                    margin-top: 8px; 
                    margin-bottom: 8px; 
                    padding-left: 25px;
                }}
                li {{ 
                    margin-bottom: 5px; 
                }}
                .page-break {{ 
                    page-break-before: always; 
                }}
                strong {{
                    color: #2c3e50;
                }}
                .toc {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    margin-bottom: 25px;
                    border-radius: 5px;
                }}
                .toc h2 {{
                    margin-top: 0;
                    color: #3498db;
                    border-bottom: none;
                }}
                .toc ul {{
                    list-style-type: none;
                    padding-left: 10px;
                }}
                .toc-h1 {{
                    font-weight: bold;
                    margin-top: 8px;
                }}
                .toc-h2 {{
                    padding-left: 15px;
                    margin-top: 5px;
                }}
                .toc-h3 {{
                    padding-left: 30px;
                    font-size: 10pt;
                }}
                .executive-summary {{
                    background-color: #e8f4fc;
                    padding: 12px;
                    border-left: 4px solid #3498db;
                    margin: 20px 0;
                }}
                .key-insight {{
                    background-color: #f5f5f5;
                    padding: 10px;
                    margin: 12px 0;
                    border-left: 3px solid #27ae60;
                }}
                .recommendation {{
                    background-color: #eafaf1;
                    padding: 10px;
                    margin: 15px 0;
                    border-left: 3px solid #2ecc71;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Create HTML file
        html_path = os.path.join(output_dir, "report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(styled_html)
        
        # Generate PDF from HTML using xhtml2pdf
        pdf_path = os.path.join(output_dir, "report.pdf")
        
        with open(pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                styled_html,               # HTML content
                dest=pdf_file,             # Output file
                encoding='utf-8'
            )
        
        if pisa_status.err:
            self.logger.error('Error converting HTML to PDF')
            return None
            
        return pdf_path
    
    def generate_visualizations(self, df: pd.DataFrame, insights: Dict[str, Any], 
                            output_dir: str) -> Dict[str, Any]:
        """Generate user-friendly visualizations with better analysis focus."""
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        viz_paths = {}
        errors = []
        
        # Set consistent styling for all visualizations
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        # 1. Enhanced Data Overview
        try:
            # Summary statistics visualization
            numeric_cols = df.select_dtypes(include=['number']).columns[:5]  # First 5 numeric columns
            if len(numeric_cols) > 0:
                try:
                    # Create a summary statistics table visualized as a heatmap
                    summary = df[numeric_cols].describe().T
                    plt.figure(figsize=(8, 5))  # REDUCED SIZE from (12, 8)
                    sns.heatmap(summary, annot=True, fmt='.2f', cmap='Blues', linewidths=.5)
                    plt.title('Summary Statistics', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    path = os.path.join(viz_dir, "summary_stats.png")
                    plt.savefig(path, bbox_inches='tight', dpi=150)
                    plt.close()
                    viz_paths["Data Summary"] = path
                except Exception as e:
                    errors.append(f"Summary statistics visualization failed: {str(e)}")
        
            # 2. Enhanced numeric distributions with better styling
            for i, col in enumerate(numeric_cols[:3]):  # First 3 numeric columns
                try:
                    plt.figure(figsize=(8, 5))  # REDUCED SIZE from (12, 8)
                    # Create a more attractive distribution plot
                    ax = sns.histplot(df[col], kde=True, color=colors[i % len(colors)])
                    
                    # Add mean and median lines
                    mean_val = df[col].mean()
                    median_val = df[col].median()
                    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                            label=f'Mean: {mean_val:.2f}')
                    ax.axvline(median_val, color='green', linestyle='-.', linewidth=1.5, 
                            label=f'Median: {median_val:.2f}')
                    
                    # Add stats annotation
                    textstr = f'Standard Dev: {df[col].std():.2f}\nRange: {df[col].min():.2f} - {df[col].max():.2f}'
                    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
                    
                    plt.title(f"Distribution of {col}", fontsize=14, fontweight='bold')
                    plt.xlabel(col, fontsize=12)
                    plt.ylabel("Frequency", fontsize=12)
                    plt.legend(fontsize=10)
                    plt.tight_layout()
                    
                    path = os.path.join(viz_dir, f"dist_{col}.png")
                    plt.savefig(path, bbox_inches='tight', dpi=150)
                    plt.close()
                    viz_paths[f"Distribution of {col}"] = path
                except Exception as e:
                    errors.append(f"Distribution plot failed for {col}: {str(e)}")
            
            # 3. Enhanced correlation heatmap
            if len(numeric_cols) > 1:
                try:
                    plt.figure(figsize=(8, 7))  # REDUCED SIZE from (12, 10)
                    corr = df[numeric_cols].corr()
                    mask = np.triu(np.ones_like(corr, dtype=bool))  # Create mask for upper triangle
                    
                    # Generate a custom diverging colormap
                    cmap = sns.diverging_palette(230, 20, as_cmap=True)
                    
                    # Draw the heatmap with the mask and correct aspect ratio
                    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                            annot=True, fmt=".2f", square=True, linewidths=.5, annot_kws={"size": 12})
                    
                    plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    path = os.path.join(viz_dir, "correlation.png")
                    plt.savefig(path, bbox_inches='tight', dpi=150)
                    plt.close()
                    viz_paths["Correlation Matrix"] = path
                except Exception as e:
                    errors.append(f"Correlation heatmap failed: {str(e)}")
            
            # 4. Categorical data visualization (if available)
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns[:3]
            for i, col in enumerate(categorical_cols):
                try:
                    value_counts = df[col].value_counts().head(10)  # Top 10 categories
                    plt.figure(figsize=(9, 6))  # REDUCED SIZE from (14, 10)
                    ax = value_counts.plot(kind='bar', color=colors[i % len(colors)])
                    
                    # Add count labels on top of bars
                    for j, v in enumerate(value_counts):
                        ax.text(j, v + 0.1, str(v), ha='center', fontweight='bold', fontsize=10)
                    
                    plt.title(f"Top Categories in {col}", fontsize=14, fontweight='bold')
                    plt.xlabel(col, fontsize=12)
                    plt.ylabel("Count", fontsize=12)
                    plt.xticks(rotation=45, ha='right', fontsize=10)
                    plt.tight_layout()
                    
                    path = os.path.join(viz_dir, f"cat_{col}.png")
                    plt.savefig(path, bbox_inches='tight', dpi=150)
                    plt.close()
                    viz_paths[f"Categories in {col}"] = path
                except Exception as e:
                    errors.append(f"Categorical plot failed for {col}: {str(e)}")
        
        except Exception as e:
            errors.append(f"Standard visualization failed: {str(e)}")
        
        # 5. Insight-specific visualizations with better styling
        try:
            # Outliers with improved visualization
            if "outliers" in insights and "detected" in insights["outliers"]:
                for i, (col, _) in enumerate(list(insights["outliers"]["detected"].items())[:3]):
                    if col in df.columns:
                        try:
                            plt.figure(figsize=(9, 6))  # REDUCED SIZE from (14, 10)
                            # Create boxplot with swarmplot overlay for better visualization
                            ax = sns.boxplot(x=df[col], color=colors[i % len(colors)], width=0.5)
                            
                            # Calculate outlier thresholds
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            # Add annotations
                            textstr = f'Q1: {Q1:.2f}\nMedian: {df[col].median():.2f}\nQ3: {Q3:.2f}\n'
                            textstr += f'Lower bound: {lower_bound:.2f}\nUpper bound: {upper_bound:.2f}'
                            props = dict(boxstyle='round', facecolor='white', alpha=0.7)
                            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                                verticalalignment='top', bbox=props)
                            
                            plt.title(f"Outlier Analysis: {col}", fontsize=14, fontweight='bold')
                            plt.tight_layout()
                            
                            path = os.path.join(viz_dir, f"outliers_{col}.png")
                            plt.savefig(path, bbox_inches='tight', dpi=150)
                            plt.close()
                            viz_paths[f"Outliers in {col}"] = path
                        except Exception as e:
                            errors.append(f"Outlier plot failed for {col}: {str(e)}")
            
            # Feature importance with enhanced visualization
            if "predictive" in insights and "feature_importance" in insights["predictive"]:
                try:
                    features = insights["predictive"]["feature_importance"]
                    plt.figure(figsize=(9, 6))  # REDUCED SIZE from (14, 10)
                    
                    # Convert to Series and sort
                    feature_imp = pd.Series(features).sort_values(ascending=True)
                    
                    # Create horizontal bar plot with gradient colors
                    bars = plt.barh(range(len(feature_imp)), feature_imp, color=plt.cm.viridis(np.linspace(0.1, 0.9, len(feature_imp))))
                    
                    # Customize axes
                    plt.yticks(range(len(feature_imp)), feature_imp.index, fontsize=12)
                    plt.xlabel('Importance Score', fontsize=12)
                    plt.title('Feature Importance', fontsize=14, fontweight='bold')
                    
                    # Add value labels
                    for i, v in enumerate(feature_imp):
                        plt.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=10)
                    
                    plt.tight_layout()
                    path = os.path.join(viz_dir, "feature_importance.png")
                    plt.savefig(path, bbox_inches='tight', dpi=150)
                    plt.close()
                    viz_paths["Feature Importance"] = path
                except Exception as e:
                    errors.append(f"Feature importance failed: {str(e)}")
        
        except Exception as e:
            errors.append(f"Insight visualization failed: {str(e)}")
        
        return {
            "paths": viz_paths,
            "errors": errors,
            "status": "success" if not errors else "partial"
        }

    def _generate_report_content(self, insights: Dict[str, Any], 
                       visualization_paths: Dict[str, str],
                       dataset_name: str, 
                       target_column: Optional[str]) -> str:
        """Generate user-friendly markdown report content with enhanced analysis."""
        # Prepare visualization references
        viz_refs = "\n".join(
            f"- {name}: ![]({path})" 
            for name, path in visualization_paths.items()
        )
        
        prompt = f"""
        Generate an insightful, business-focused data analysis report in Markdown format.
        Focus on clear, actionable insights rather than just describing the data.
        Each finding should connect to potential business impact or decision-making.
        
        **DO NOT WRITE MARKDOWNN AT THE TOP**
        
        Dataset: {dataset_name}
        {f"Target Variable: {target_column}" if target_column else ""}
        
        Available Visualizations:
        {viz_refs if viz_refs else "No visualizations available"}
        
        Insights:
        {json.dumps(insights, indent=2)}
        
        Report Structure:
        1. Executive Summary (1-2 paragraphs highlighting business impact of key findings)
        2. Data Context (brief contextual overview, NOT just statistics)
        3. Key Insights (3-5 most important findings with clear business implications)
        4. Visual Analysis (for each visualization, focus on the "so what" not just "what")
        5. Strategic Recommendations (specific, actionable next steps tied to insights)
        6. Technical Appendix (brief, for data specialists only)
        
        For each insight and visualization:
        - Focus on WHY this matters, not just WHAT the data shows
        - Connect insights to potential business outcomes or decisions
        - Highlight anomalies, trends, or patterns that might affect business strategy
        - Suggest specific actions that could be taken based on these insights
        
        Format guidelines:
        - Include a table of contents with links to each section
        - Use lists and bullet points for key takeaways
        - Bold important findings or statistics
        - Keep paragraphs short (3-4 sentences max)
        - Include all visualization references as ![](path/to/image.png)
        - Use business terminology rather than technical jargon
        """
        
        response = self.model.generate_content(prompt)
        return response.text