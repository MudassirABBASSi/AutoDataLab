"""
Report generation module for AutoDataLab.
Generates comprehensive analysis reports in multiple formats.
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from io import StringIO, BytesIO
import tempfile
import os

# Lazy imports for reportlab - only imported when PDF generation is needed
# from reportlab.lib.pagesizes import letter, A4
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# etc.

from utils.logger import get_logger
from utils.exceptions import ReportingError
from core.data_validator import DataValidator

logger = get_logger(__name__)


class ReportGenerator:
    """Generate professional reports from analysis results."""
    
    def __init__(self, title: str = "AutoDataLab Report"):
        """
        Initialize report generator.
        
        Args:
            title: Report title
        """
        self.title = title
        self.sections: Dict[str, Any] = {}
        self.generated_at = datetime.now()
        logger.info(f"Report generator initialized: {title}")
    
    def add_section(self, section_name: str, content: Any) -> None:
        """
        Add a section to the report.
        
        Args:
            section_name: Name of section
            content: Section content
        """
        self.sections[section_name] = content
        logger.debug(f"Added section: {section_name}")
    
    def add_dataframe_section(
        self,
        section_name: str,
        df: pd.DataFrame,
        description: str = ""
    ) -> None:
        """
        Add DataFrame section to report.
        
        Args:
            section_name: Section name
            df: DataFrame to include
            description: Section description
        """
        try:
            section_data = {
                "description": description,
                "rows": len(df),
                "columns": len(df.columns),
                "data": df.head(10).to_dict('records'),
                "columns_list": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
            self.add_section(section_name, section_data)
            logger.debug(f"Added DataFrame section: {section_name}")
        except Exception as e:
            logger.error(f"Error adding DataFrame section: {e}")
            raise ReportingError(f"Failed to add DataFrame section: {e}")
    
    def add_statistics_section(
        self,
        section_name: str,
        df: pd.DataFrame,
        description: str = ""
    ) -> None:
        """
        Add statistics section to report.
        
        Args:
            section_name: Section name
            df: DataFrame to analyze
            description: Section description
        """
        try:
            numeric_df = df.select_dtypes(include=['number'])
            
            section_data = {
                "description": description,
                "statistics": numeric_df.describe().to_dict(),
                "correlation": numeric_df.corr().to_dict() if not numeric_df.empty else {},
                "missing_values": df.isna().sum().to_dict()
            }
            self.add_section(section_name, section_data)
            logger.debug(f"Added statistics section: {section_name}")
        except Exception as e:
            logger.error(f"Error adding statistics section: {e}")
            raise ReportingError(f"Failed to add statistics section: {e}")
    
    def add_data_quality_section(
        self,
        section_name: str,
        df: pd.DataFrame
    ) -> None:
        """
        Add data quality report section.
        
        Args:
            section_name: Section name
            df: DataFrame to assess
        """
        try:
            quality_report = DataValidator.get_comprehensive_report(df)
            self.add_section(section_name, quality_report)
            logger.debug(f"Added data quality section: {section_name}")
        except Exception as e:
            logger.error(f"Error adding data quality section: {e}")
            raise ReportingError(f"Failed to add data quality section: {e}")
    
    def add_model_results_section(
        self,
        section_name: str,
        model_name: str,
        metrics: Dict[str, float],
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add model results section.
        
        Args:
            section_name: Section name
            model_name: Name of model
            metrics: Model evaluation metrics
            additional_info: Additional model information
        """
        try:
            section_data = {
                "model": model_name,
                "metrics": metrics,
                "additional_info": additional_info or {}
            }
            self.add_section(section_name, section_data)
            logger.debug(f"Added model results section: {section_name}")
        except Exception as e:
            logger.error(f"Error adding model results section: {e}")
            raise ReportingError(f"Failed to add model results section: {e}")
    
    def add_summary(self, key: str, value: Any) -> None:
        """
        Add summary key-value pair.
        
        Args:
            key: Summary key
            value: Summary value
        """
        if "summary" not in self.sections:
            self.sections["summary"] = {}
        
        self.sections["summary"][key] = value
        logger.debug(f"Added summary: {key}")
    
    def generate_html(self) -> str:
        """
        Generate HTML report.
        
        Returns:
            str: HTML content
        """
        try:
            html_parts = []
            
            # Header
            html_parts.append(f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{self.title}</title>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    .header {{
                        background-color: #1F3A8A;
                        color: white;
                        padding: 30px;
                        border-radius: 8px;
                        margin-bottom: 30px;
                    }}
                    .header h1 {{
                        margin: 0;
                        font-size: 2.5em;
                    }}
                    .header p {{
                        margin: 10px 0 0 0;
                        opacity: 0.9;
                    }}
                    .section {{
                        background-color: white;
                        padding: 20px;
                        margin-bottom: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .section h2 {{
                        color: #1F3A8A;
                        border-bottom: 2px solid #1F3A8A;
                        padding-bottom: 10px;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 15px 0;
                    }}
                    table th {{
                        background-color: #f0f0f0;
                        padding: 10px;
                        text-align: left;
                        font-weight: bold;
                    }}
                    table td {{
                        padding: 8px;
                        border-bottom: 1px solid #ddd;
                    }}
                    table tr:hover {{
                        background-color: #f9f9f9;
                    }}
                    .metric {{
                        display: inline-block;
                        background-color: #f0f0f0;
                        padding: 15px;
                        margin: 10px;
                        border-radius: 5px;
                        border-left: 4px solid #1F3A8A;
                    }}
                    .metric .value {{
                        font-size: 1.5em;
                        font-weight: bold;
                        color: #1F3A8A;
                    }}
                    .metric .label {{
                        font-size: 0.9em;
                        color: #666;
                    }}
                    .footer {{
                        text-align: center;
                        color: #999;
                        font-size: 0.9em;
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{self.title}</h1>
                    <p>Generated on {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """)
            
            # Summary section
            if "summary" in self.sections:
                html_parts.append('<div class="section"><h2>Summary</h2>')
                for key, value in self.sections["summary"].items():
                    if isinstance(value, (int, float)):
                        html_parts.append(f"""
                        <div class="metric">
                            <div class="label">{key}</div>
                            <div class="value">{value}</div>
                        </div>
                        """)
                    else:
                        html_parts.append(f'<p><strong>{key}:</strong> {value}</p>')
                html_parts.append('</div>')
            
            # Other sections
            for section_name, content in self.sections.items():
                if section_name == "summary":
                    continue
                
                html_parts.append(f'<div class="section"><h2>{section_name}</h2>')
                html_parts.append(self._format_content_html(content))
                html_parts.append('</div>')
            
            # Footer
            html_parts.append("""
                <div class="footer">
                    <p>This report was automatically generated by AutoDataLab</p>
                </div>
            </body>
            </html>
            """)
            
            logger.info("HTML report generated")
            return "\n".join(html_parts)
        
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise ReportingError(f"Failed to generate HTML report: {e}")
    
    def generate_json(self) -> str:
        """
        Generate JSON report.
        
        Returns:
            str: JSON content
        """
        try:
            report_data = {
                "title": self.title,
                "generated_at": self.generated_at.isoformat(),
                "sections": self._convert_to_serializable(self.sections)
            }
            logger.info("JSON report generated")
            return json.dumps(report_data, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            raise ReportingError(f"Failed to generate JSON report: {e}")
    
    def generate_text(self) -> str:
        """
        Generate text report.
        
        Returns:
            str: Text content
        """
        try:
            text_parts = []
            
            # Header
            text_parts.append("=" * 80)
            text_parts.append(self.title.center(80))
            text_parts.append("=" * 80)
            text_parts.append(f"\nGenerated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            # Sections
            for section_name, content in self.sections.items():
                text_parts.append("\n" + "-" * 80)
                text_parts.append(f"{section_name.upper()}")
                text_parts.append("-" * 80)
                text_parts.append(self._format_content_text(content))
            
            text_parts.append("\n" + "=" * 80)
            text_parts.append("End of Report".center(80))
            text_parts.append("=" * 80)
            
            logger.info("Text report generated")
            return "\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error generating text report: {e}")
            raise ReportingError(f"Failed to generate text report: {e}")
    
    def save_report(
        self,
        filename: str,
        format: str = "html",
        output_dir: Optional[str] = None
    ) -> str:
        """
        Save report to file.
        
        Args:
            filename: Output filename (without extension)
            format: Format ('html', 'json', or 'text')
            output_dir: Output directory
            
        Returns:
            str: Path to saved file
        """
        try:
            if format == "html":
                content = self.generate_html()
                ext = ".html"
            elif format == "json":
                content = self.generate_json()
                ext = ".json"
            elif format == "text":
                content = self.generate_text()
                ext = ".txt"
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            if output_dir is None:
                output_dir = "."
            
            output_path = Path(output_dir) / f"{filename}{ext}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Report saved: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise ReportingError(f"Failed to save report: {e}")
    
    @staticmethod
    def _format_content_html(content: Any) -> str:
        """Format content as HTML."""
        if isinstance(content, dict):
            html = "<table>"
            for key, value in content.items():
                if isinstance(value, dict):
                    html += f"<tr><td><strong>{key}</strong></td><td>"
                    html += ReportGenerator._format_content_html(value)
                    html += "</td></tr>"
                elif isinstance(value, list):
                    html += f"<tr><td><strong>{key}</strong></td><td>{len(value)} items</td></tr>"
                else:
                    html += f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>"
            html += "</table>"
            return html
        else:
            return f"<p>{content}</p>"
    
    @staticmethod
    def _format_content_text(content: Any, indent: int = 0) -> str:
        """Format content as text."""
        if isinstance(content, dict):
            lines = []
            for key, value in content.items():
                if isinstance(value, dict):
                    lines.append(f"{'  ' * indent}{key}:")
                    lines.append(ReportGenerator._format_content_text(value, indent + 1))
                elif isinstance(value, list):
                    lines.append(f"{'  ' * indent}{key}: {len(value)} items")
                else:
                    lines.append(f"{'  ' * indent}{key}: {value}")
            return "\n".join(lines)
        else:
            return f"{'  ' * indent}{content}"
    
    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: ReportGenerator._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ReportGenerator._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        else:
            return str(obj)


def generate_eda_report(
    df: pd.DataFrame,
    model: Any = None,
    model_metrics: Optional[Dict[str, float]] = None,
    feature_importance: Optional[pd.DataFrame] = None,
    output_dir: str = "reports",
    filename: str = None
) -> str:
    """
    Generate comprehensive EDA PDF report with model performance.
    
    Args:
        df: Input DataFrame
        model: Trained model (optional)
        model_metrics: Model evaluation metrics (optional)
        feature_importance: Feature importance DataFrame (optional)
        output_dir: Output directory for report
        filename: Custom filename (default: AutoDataLab_Report_YYYYMMDD_HHMMSS.pdf)
        
    Returns:
        str: Path to generated PDF report
        
    Raises:
        ReportingError: If report generation fails
    """
    try:
        # Import reportlab components (lazy import)
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, Image as RLImage, KeepTogether
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        
        logger.info("Starting PDF report generation")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"AutoDataLab_Report_{timestamp}.pdf"
        
        if not filename.endswith('.pdf'):
            filename += '.pdf'
            
        report_path = output_path / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Container for PDF elements
        elements = []
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1F3A8A'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1F3A8A'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold',
            borderColor=colors.HexColor('#1F3A8A'),
            borderWidth=2,
            borderPadding=5,
            backColor=colors.HexColor('#F0F4FF')
        )
        
        heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#334155'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=10,
            spaceAfter=8,
            fontName='Helvetica'
        )
        
        # SECTION 1: Cover Page
        logger.debug("Creating cover page")
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("AutoDataLab", title_style))
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("Automated Data Science Report", styles['Heading2']))
        elements.append(Spacer(1, 0.5*inch))
        
        # Dataset shape
        dataset_info = f"""
        <para align=center>
        <b>Dataset Shape:</b> {df.shape[0]:,} rows × {df.shape[1]} columns<br/>
        <b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}
        </para>
        """
        elements.append(Paragraph(dataset_info, body_style))
        elements.append(PageBreak())
        
        # SECTION 2: Dataset Summary
        logger.debug("Creating dataset summary")
        elements.append(Paragraph("1. Dataset Summary", heading1_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Data types table
        elements.append(Paragraph("Data Types Distribution", heading2_style))
        dtype_counts = df.dtypes.value_counts()
        dtype_data = [['Data Type', 'Count']]
        for dtype, count in dtype_counts.items():
            dtype_data.append([str(dtype), str(count)])
        
        dtype_table = Table(dtype_data, colWidths=[3*inch, 2*inch])
        dtype_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A8A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
        ]))
        elements.append(dtype_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Memory usage
        memory_usage = df.memory_usage(deep=True).sum()
        memory_mb = memory_usage / (1024 * 1024)
        elements.append(Paragraph(f"<b>Memory Usage:</b> {memory_mb:.2f} MB", body_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # First 5 rows preview
        elements.append(Paragraph("Data Preview (First 5 Rows)", heading2_style))
        preview_df = df.head(5)
        
        # Limit columns to fit on page
        max_cols = 6
        if len(preview_df.columns) > max_cols:
            preview_df = preview_df.iloc[:, :max_cols]
            col_note = f"<i>Showing {max_cols} of {len(df.columns)} columns</i>"
            elements.append(Paragraph(col_note, body_style))
        
        preview_data = [list(preview_df.columns)]
        for idx, row in preview_df.iterrows():
            preview_data.append([str(val)[:30] + '...' if len(str(val)) > 30 else str(val) 
                               for val in row])
        
        col_width = 6.5 * inch / len(preview_df.columns)
        preview_table = Table(preview_data, colWidths=[col_width] * len(preview_df.columns))
        preview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A8A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
        ]))
        elements.append(preview_table)
        elements.append(PageBreak())
        
        # SECTION 3: Missing Values Analysis
        logger.debug("Creating missing values analysis")
        elements.append(Paragraph("2. Missing Values Analysis", heading1_style))
        elements.append(Spacer(1, 0.2*inch))
        
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            # Missing values table
            missing_data = [['Column', 'Missing Count', 'Percentage']]
            for _, row in missing_df.iterrows():
                missing_data.append([
                    row['Column'],
                    str(row['Missing Count']),
                    f"{row['Percentage']:.2f}%"
                ])
            
            missing_table = Table(missing_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            missing_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A8A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
            ]))
            
            # Highlight rows with >20% missing
            for i, row in enumerate(missing_df.iterrows(), start=1):
                if row[1]['Percentage'] > 20:
                    missing_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#FEE2E2')),
                        ('TEXTCOLOR', (0, i), (-1, i), colors.HexColor('#991B1B'))
                    ]))
            
            elements.append(missing_table)
            elements.append(Spacer(1, 0.2*inch))
            
            # Missing values bar plot
            if len(missing_df) <= 15:  # Only plot if reasonable number
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.barh(missing_df['Column'], missing_df['Percentage'])
                
                # Color bars based on threshold
                for i, (_, row) in enumerate(missing_df.iterrows()):
                    if row['Percentage'] > 20:
                        bars[i].set_color('#EF4444')
                    else:
                        bars[i].set_color('#1F3A8A')
                
                ax.set_xlabel('Missing Percentage (%)', fontsize=11, fontweight='bold')
                ax.set_ylabel('Column', fontsize=11, fontweight='bold')
                ax.set_title('Missing Values by Column', fontsize=13, fontweight='bold')
                ax.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='20% threshold')
                ax.legend()
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                # Save plot
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                plt.close(fig)
                
                img = RLImage(img_buffer, width=5.5*inch, height=3.5*inch)
                elements.append(img)
        else:
            elements.append(Paragraph("<b>No missing values detected in the dataset.</b>", body_style))
        
        elements.append(PageBreak())
        
        # SECTION 4: Distribution Analysis
        logger.debug("Creating distribution analysis")
        elements.append(Paragraph("3. Distribution Analysis", heading1_style))
        elements.append(Spacer(1, 0.2*inch))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            # Select top 5 numeric columns (by variance or first 5)
            top_numeric = numeric_cols[:min(5, len(numeric_cols))]
            
            for col in top_numeric:
                elements.append(Paragraph(f"Distribution: {col}", heading2_style))
                
                # Create histogram and boxplot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
                
                # Histogram
                ax1.hist(df[col].dropna(), bins=30, color='#1F3A8A', alpha=0.7, edgecolor='black')
                ax1.set_xlabel(col, fontweight='bold')
                ax1.set_ylabel('Frequency', fontweight='bold')
                ax1.set_title(f'{col} - Histogram', fontweight='bold')
                ax1.grid(axis='y', alpha=0.3)
                
                # Boxplot
                ax2.boxplot(df[col].dropna(), vert=True)
                ax2.set_ylabel(col, fontweight='bold')
                ax2.set_title(f'{col} - Boxplot', fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                img_buffer = BytesIO()
                plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                plt.close(fig)
                
                img = RLImage(img_buffer, width=6.5*inch, height=2*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.15*inch))
                
                # Statistics
                stats_text = f"""
                <b>Mean:</b> {df[col].mean():.2f} | 
                <b>Median:</b> {df[col].median():.2f} | 
                <b>Std:</b> {df[col].std():.2f} | 
                <b>Min:</b> {df[col].min():.2f} | 
                <b>Max:</b> {df[col].max():.2f}
                """
                elements.append(Paragraph(stats_text, body_style))
                elements.append(Spacer(1, 0.2*inch))
        else:
            elements.append(Paragraph("<b>No numeric columns found for distribution analysis.</b>", body_style))
        
        elements.append(PageBreak())
        
        # SECTION 5: Correlation Matrix
        logger.debug("Creating correlation analysis")
        elements.append(Paragraph("4. Correlation Analysis", heading1_style))
        elements.append(Spacer(1, 0.2*inch))
        
        if len(numeric_cols) >= 2:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                corr_matrix,
                annot=len(numeric_cols) <= 10,  # Annotate if not too many columns
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
            ax.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save plot
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close(fig)
            
            img = RLImage(img_buffer, width=5.5*inch, height=4.5*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
            
            # Top correlations
            elements.append(Paragraph("Top 5 Highly Correlated Feature Pairs", heading2_style))
            
            # Get correlation pairs
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        abs(corr_matrix.iloc[i, j]),
                        corr_matrix.iloc[i, j]
                    ))
            
            corr_pairs.sort(key=lambda x: x[2], reverse=True)
            top_corr = corr_pairs[:5]
            
            corr_data = [['Feature 1', 'Feature 2', 'Correlation']]
            for feat1, feat2, abs_corr, corr in top_corr:
                corr_data.append([feat1, feat2, f"{corr:.3f}"])
            
            corr_table = Table(corr_data, colWidths=[2.5*inch, 2.5*inch, 1.5*inch])
            corr_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A8A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
            ]))
            elements.append(corr_table)
        else:
            elements.append(Paragraph("<b>Insufficient numeric columns for correlation analysis.</b>", body_style))
        
        elements.append(PageBreak())
        
        # SECTION 6: Feature Importance (if available)
        if feature_importance is not None and len(feature_importance) > 0:
            logger.debug("Creating feature importance section")
            elements.append(Paragraph("5. Feature Importance", heading1_style))
            elements.append(Spacer(1, 0.2*inch))
            
            # Get top 10 features
            top_features = feature_importance.head(10)
            
            # Create bar plot
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(top_features.iloc[:, 0], top_features.iloc[:, 1], color='#1F3A8A')
            ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
            ax.set_ylabel('Feature', fontsize=11, fontweight='bold')
            ax.set_title('Top 10 Important Features', fontsize=13, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close(fig)
            
            img = RLImage(img_buffer, width=5.5*inch, height=3.5*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
            
            # Feature importance table
            feat_data = [['Rank', 'Feature', 'Importance']]
            for i, (_, row) in enumerate(top_features.iterrows(), start=1):
                feat_data.append([str(i), str(row.iloc[0]), f"{row.iloc[1]:.4f}"])
            
            feat_table = Table(feat_data, colWidths=[1*inch, 3*inch, 2*inch])
            feat_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A8A')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
            ]))
            elements.append(feat_table)
            elements.append(PageBreak())
        
        # SECTION 7: Model Performance (if available)
        if model is not None and model_metrics is not None:
            logger.debug("Creating model performance section")
            section_num = "6" if feature_importance is None else "6"
            elements.append(Paragraph(f"{section_num}. Model Performance Summary", heading1_style))
            elements.append(Spacer(1, 0.2*inch))
            
            # Determine if classification or regression
            is_classification = any(key in model_metrics for key in ['accuracy', 'precision', 'recall', 'f1'])
            
            if is_classification:
                elements.append(Paragraph("Classification Metrics", heading2_style))
                
                # Metrics table
                metrics_data = [['Metric', 'Value']]
                for metric, value in model_metrics.items():
                    if isinstance(value, (int, float)) and metric != 'confusion_matrix':
                        metrics_data.append([metric.replace('_', ' ').title(), f"{value:.4f}"])
                
                metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A8A')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
                ]))
                elements.append(metrics_table)
                elements.append(Spacer(1, 0.2*inch))
                
                # Confusion matrix (if available)
                if 'confusion_matrix' in model_metrics:
                    elements.append(Paragraph("Confusion Matrix", heading2_style))
                    cm = model_metrics['confusion_matrix']
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
                    ax.set_xlabel('Predicted', fontweight='bold')
                    ax.set_ylabel('Actual', fontweight='bold')
                    ax.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
                    plt.tight_layout()
                    
                    img_buffer = BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    plt.close(fig)
                    
                    img = RLImage(img_buffer, width=4.5*inch, height=4*inch)
                    elements.append(img)
                
                # ROC curve (if available)
                if 'roc_curve' in model_metrics:
                    elements.append(Spacer(1, 0.2*inch))
                    elements.append(Paragraph("ROC Curve", heading2_style))
                    
                    fpr, tpr, _ = model_metrics['roc_curve']
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.plot(fpr, tpr, color='#1F3A8A', lw=2, label=f'ROC Curve')
                    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
                    ax.set_xlabel('False Positive Rate', fontweight='bold')
                    ax.set_ylabel('True Positive Rate', fontweight='bold')
                    ax.set_title('ROC Curve', fontsize=13, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    
                    img_buffer = BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    plt.close(fig)
                    
                    img = RLImage(img_buffer, width=4.5*inch, height=4*inch)
                    elements.append(img)
                    
            else:  # Regression
                elements.append(Paragraph("Regression Metrics", heading2_style))
                
                # Metrics table
                metrics_data = [['Metric', 'Value']]
                for metric, value in model_metrics.items():
                    if isinstance(value, (int, float)) and metric not in ['y_true', 'y_pred']:
                        metrics_data.append([metric.replace('_', ' ').upper(), f"{value:.4f}"])
                
                metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
                metrics_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1F3A8A')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
                ]))
                elements.append(metrics_table)
                elements.append(Spacer(1, 0.2*inch))
                
                # Prediction vs Actual scatter plot (if available)
                if 'y_true' in model_metrics and 'y_pred' in model_metrics:
                    elements.append(Paragraph("Prediction vs Actual", heading2_style))
                    
                    y_true = model_metrics['y_true']
                    y_pred = model_metrics['y_pred']
                    
                    # Limit points for performance
                    if len(y_true) > 1000:
                        indices = np.random.choice(len(y_true), 1000, replace=False)
                        y_true = y_true[indices]
                        y_pred = y_pred[indices]
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(y_true, y_pred, alpha=0.5, color='#1F3A8A', s=30)
                    
                    # Perfect prediction line
                    min_val = min(y_true.min(), y_pred.min())
                    max_val = max(y_true.max(), y_pred.max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
                    
                    ax.set_xlabel('Actual Values', fontweight='bold')
                    ax.set_ylabel('Predicted Values', fontweight='bold')
                    ax.set_title('Predicted vs Actual Values', fontsize=13, fontweight='bold')
                    ax.legend()
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    
                    img_buffer = BytesIO()
                    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                    img_buffer.seek(0)
                    plt.close(fig)
                    
                    img = RLImage(img_buffer, width=5*inch, height=4*inch)
                    elements.append(img)
        
        # Footer
        elements.append(PageBreak())
        elements.append(Spacer(1, 3*inch))
        footer_text = """
        <para align=center>
        <b>End of Report</b><br/>
        <i>This report was automatically generated by AutoDataLab</i><br/>
        {}<br/>
        © 2026 AutoDataLab - Automated Data Science Platform
        </para>
        """.format(datetime.now().strftime('%B %d, %Y at %H:%M:%S'))
        elements.append(Paragraph(footer_text, body_style))
        
        # Build PDF
        logger.info("Building PDF document")
        doc.build(elements)
        
        logger.info(f"PDF report generated successfully: {report_path}")
        return str(report_path)
        
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}", exc_info=True)
        raise ReportingError(f"Failed to generate PDF report: {e}")


if __name__ == "__main__":
    # Test report generation
    logger.info("Testing report generation")
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': ['a', 'b', 'c', 'd', 'e']
    })
    
    try:
        # Create report
        report = ReportGenerator("Sample Report")
        report.add_summary("Dataset Name", "Sample Dataset")
        report.add_summary("Records", len(df))
        report.add_dataframe_section("Input Data", df, "Sample data")
        report.add_statistics_section("Statistics", df)
        
        # Generate text report
        text_report = report.generate_text()
        print("✓ Text report generated")
        print("\n" + text_report[:500] + "...")
    
    except Exception as e:
        print(f"✗ Error: {e}")
