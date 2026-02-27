"""
Plot generation utilities for AutoDataLab.
Provides centralized plotting functions with consistent styling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Any
from pathlib import Path

from utils.logger import get_logger
from utils.exceptions import VisualizationError
from config import settings

logger = get_logger(__name__)


class PlotGenerator:
    """Centralized plot generator with consistent styling."""
    
    # Color palette
    COLORS = {
        "primary": "#1F3A8A",
        "secondary": "#334155",
        "accent": "#0F766E",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "neutral": "#6B7280"
    }
    
    @staticmethod
    def set_style() -> None:
        """Apply consistent styling to plots."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = settings.DEFAULT_FIGSIZE
        plt.rcParams['font.size'] = 10
        logger.debug("Plot style applied")
    
    @staticmethod
    def histogram(
        data: pd.Series,
        title: str = "Distribution",
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        bins: int = settings.HISTOGRAM_BINS,
        color: str = "primary"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create histogram plot.
        
        Args:
            data: Series with data to plot
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            bins: Number of bins
            color: Color name from COLORS dict
            
        Returns:
            tuple: (figure, axes)
        """
        try:
            PlotGenerator.set_style()
            fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)
            
            ax.hist(
                data.dropna(),
                bins=bins,
                color=PlotGenerator.COLORS[color],
                alpha=0.7,
                edgecolor='black'
            )
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', alpha=0.3)
            
            logger.debug(f"Histogram created: {title}")
            return fig, ax
        
        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            raise VisualizationError(f"Failed to create histogram: {e}")
    
    @staticmethod
    def box_plot(
        data: pd.DataFrame,
        columns: List[str],
        title: str = "Box Plot",
        ylabel: str = "Value",
        color: str = "primary"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create box plot.
        
        Args:
            data: DataFrame with data
            columns: Columns to plot
            title: Plot title
            ylabel: Y-axis label
            color: Color palette name
            
        Returns:
            tuple: (figure, axes)
        """
        try:
            PlotGenerator.set_style()
            fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)
            
            plot_data = [data[col].dropna() for col in columns if col in data.columns]
            
            bp = ax.boxplot(
                plot_data,
                labels=[col for col in columns if col in data.columns],
                patch_artist=True
            )
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor(PlotGenerator.COLORS[color])
                patch.set_alpha(0.7)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            logger.debug(f"Box plot created: {title}")
            return fig, ax
        
        except Exception as e:
            logger.error(f"Error creating box plot: {e}")
            raise VisualizationError(f"Failed to create box plot: {e}")
    
    @staticmethod
    def scatter_plot(
        x_data: pd.Series,
        y_data: pd.Series,
        title: str = "Scatter Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        color: str = "primary",
        hue_data: Optional[pd.Series] = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create scatter plot.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Color name
            hue_data: Optional data for coloring points
            
        Returns:
            tuple: (figure, axes)
        """
        try:
            PlotGenerator.set_style()
            fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)
            
            if hue_data is not None:
                scatter = ax.scatter(
                    x_data.dropna(),
                    y_data.dropna(),
                    c=hue_data.dropna(),
                    cmap=settings.PLOT_COLOR_PALETTE,
                    alpha=0.6,
                    s=50,
                    edgecolors='black',
                    linewidth=0.5
                )
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(
                    x_data.dropna(),
                    y_data.dropna(),
                    color=PlotGenerator.COLORS[color],
                    alpha=0.6,
                    s=50,
                    edgecolors='black',
                    linewidth=0.5
                )
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            
            logger.debug(f"Scatter plot created: {title}")
            return fig, ax
        
        except Exception as e:
            logger.error(f"Error creating scatter plot: {e}")
            raise VisualizationError(f"Failed to create scatter plot: {e}")
    
    @staticmethod
    def correlation_heatmap(
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "coolwarm"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create correlation heatmap.
        
        Args:
            data: DataFrame with numeric columns
            title: Plot title
            figsize: Figure size
            cmap: Colormap name
            
        Returns:
            tuple: (figure, axes)
        """
        try:
            PlotGenerator.set_style()
            
            # Select only numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                raise VisualizationError("No numeric columns to plot")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            corr_matrix = numeric_data.corr()
            
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.2f',
                cmap=cmap,
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            logger.debug(f"Correlation heatmap created: {title}")
            return fig, ax
        
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {e}")
            raise VisualizationError(f"Failed to create correlation heatmap: {e}")
    
    @staticmethod
    def bar_plot(
        categories: List[str],
        values: List[float],
        title: str = "Bar Plot",
        ylabel: str = "Value",
        color: str = "primary",
        orientation: str = "vertical"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create bar plot.
        
        Args:
            categories: Category labels
            values: Values for each category
            title: Plot title
            ylabel: Y-axis label
            color: Color name
            orientation: 'vertical' or 'horizontal'
            
        Returns:
            tuple: (figure, axes)
        """
        try:
            PlotGenerator.set_style()
            fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)
            
            if orientation == "horizontal":
                ax.barh(categories, values, color=PlotGenerator.COLORS[color], alpha=0.7)
                ax.set_xlabel(ylabel)
            else:
                ax.bar(categories, values, color=PlotGenerator.COLORS[color], alpha=0.7)
                ax.set_ylabel(ylabel)
                plt.xticks(rotation=45, ha='right')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3 if orientation == "vertical" else 0.3)
            
            logger.debug(f"Bar plot created: {title}")
            return fig, ax
        
        except Exception as e:
            logger.error(f"Error creating bar plot: {e}")
            raise VisualizationError(f"Failed to create bar plot: {e}")
    
    @staticmethod
    def line_plot(
        x_data: pd.Series,
        y_data: pd.Series,
        title: str = "Line Plot",
        xlabel: str = "X",
        ylabel: str = "Y",
        color: str = "primary"
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create line plot.
        
        Args:
            x_data: X-axis data
            y_data: Y-axis data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Color name
            
        Returns:
            tuple: (figure, axes)
        """
        try:
            PlotGenerator.set_style()
            fig, ax = plt.subplots(figsize=settings.DEFAULT_FIGSIZE)
            
            ax.plot(
                x_data,
                y_data,
                color=PlotGenerator.COLORS[color],
                linewidth=2,
                marker='o',
                markersize=6,
                label='Data'
            )
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend()
            
            logger.debug(f"Line plot created: {title}")
            return fig, ax
        
        except Exception as e:
            logger.error(f"Error creating line plot: {e}")
            raise VisualizationError(f"Failed to create line plot: {e}")
    
    @staticmethod
    def save_plot(
        fig: plt.Figure,
        filename: str,
        output_dir: Optional[str] = None,
        dpi: int = 300,
        bbox_inches: str = "tight"
    ) -> str:
        """
        Save plot to file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            output_dir: Output directory (defaults to current directory)
            dpi: Resolution
            bbox_inches: Bounding box setting
            
        Returns:
            str: Path to saved file
        """
        try:
            if output_dir is None:
                output_dir = "."
            
            output_path = Path(output_dir) / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
            logger.info(f"Plot saved: {output_path}")
            
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
            raise VisualizationError(f"Failed to save plot: {e}")


if __name__ == "__main__":
    # Test plot generation
    logger.info("Testing plot generation")
    
    # Create sample data
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    
    try:
        # Test histogram
        fig, ax = PlotGenerator.histogram(df['A'], title="Sample Histogram")
        print("✓ Histogram created")
        plt.close(fig)
        
        # Test scatter plot
        fig, ax = PlotGenerator.scatter_plot(
            df['A'],
            df['B'],
            title="Sample Scatter Plot"
        )
        print("✓ Scatter plot created")
        plt.close(fig)
        
        # Test correlation heatmap
        fig, ax = PlotGenerator.correlation_heatmap(df[['A', 'B']])
        print("✓ Correlation heatmap created")
        plt.close(fig)
    
    except Exception as e:
        print(f"✗ Error: {e}")
