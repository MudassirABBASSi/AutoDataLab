"""
EDAVisualizer module for exploratory data analysis visualizations.
Provides comprehensive methods to create various statistical plots using matplotlib and seaborn.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from utils.logger import get_logger
from utils.exceptions import DataValidationError, VisualizationError

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    from statsmodels.graphics.mosaicplot import mosaic as sm_mosaic
    STATSMODELS_AVAILABLE = True
except Exception:
    sm_lowess = None
    sm_mosaic = None
    STATSMODELS_AVAILABLE = False

logger = get_logger(__name__)

# Set default seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


class EDAVisualizer:
    """
    Create comprehensive exploratory data analysis visualizations for pandas DataFrames.
    Returns matplotlib figure objects for all visualization methods.
    
    Attributes:
        df (pd.DataFrame): Copy of the original dataframe (original never modified)
        figsize (Tuple[int, int]): Default figure size for plots
    """
    
    def __init__(self, df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize EDAVisualizer with a DataFrame.
        
        Args:
            df: pandas DataFrame to visualize
            figsize: Default figure size (width, height)
            
        Raises:
            TypeError: If input is not a pandas DataFrame
            ValueError: If DataFrame is empty
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        
        if df.empty:
            raise ValueError("Cannot visualize an empty DataFrame")
        
        # Store a copy to avoid modifying original
        self.df = df.copy()
        self.figsize = figsize
        logger.info(f"EDAVisualizer initialized with shape {df.shape}")
    
    def _validate_column(self, column: str) -> None:
        """Validate that a column exists in the DataFrame."""
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
    
    def _validate_numeric(self, column: str) -> None:
        """Validate that a column is numeric."""
        self._validate_column(column)
        if not pd.api.types.is_numeric_dtype(self.df[column]):
            raise TypeError(f"Column '{column}' must be numeric, got {self.df[column].dtype}")
    
    def _validate_categorical(self, column: str) -> None:
        """Validate that a column is categorical or object type."""
        self._validate_column(column)
        if pd.api.types.is_numeric_dtype(self.df[column]) and self.df[column].nunique() > 20:
            raise TypeError(f"Column '{column}' appears to be continuous numeric, not categorical")
    
    # ==================== UNIVARIATE ANALYSIS ====================
    
    def plot_histogram(
        self,
        column: str,
        bins: int = 30,
        kde: bool = False,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a histogram for a numerical column.
        
        Args:
            column: Column name to visualize
            bins: Number of bins
            kde: Whether to overlay kernel density estimate
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If column doesn't exist
            TypeError: If column is not numerical
        """
        self._validate_numeric(column)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        data = self.df[column].dropna()
        
        ax.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7, density=kde)
        
        if kde:
            data_sorted = np.sort(data)
            kde_curve = stats.gaussian_kde(data)
            ax.plot(data_sorted, kde_curve(data_sorted), 'r-', linewidth=2, label='KDE')
            ax.legend()
        
        ax.set_xlabel(column, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density' if kde else 'Frequency', fontsize=12, fontweight='bold')
        ax.set_title(title or f'Histogram of {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        logger.info(f"Histogram created for '{column}'")
        return fig
    
    def plot_kde(
        self,
        column: str,
        fill: bool = True,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a kernel density estimate plot for a numerical column.
        
        Args:
            column: Column name to visualize
            fill: Whether to fill under the curve
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If column doesn't exist
            TypeError: If column is not numerical
        """
        self._validate_numeric(column)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        data = self.df[column].dropna()
        
        sns.kdeplot(data=data, fill=fill, color='steelblue', ax=ax, linewidth=2)
        
        ax.set_xlabel(column, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(title or f'Density Plot of {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        logger.info(f"KDE plot created for '{column}'")
        return fig
    
    def plot_boxplot(
        self,
        column: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a boxplot for a numerical column.
        
        Args:
            column: Column name to visualize
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If column doesn't exist
            TypeError: If column is not numerical
        """
        self._validate_numeric(column)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        data = self.df[column].dropna()
        
        bp = ax.boxplot([data], labels=[column], patch_artist=True, vert=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(title or f'Boxplot of {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        logger.info(f"Boxplot created for '{column}'")
        return fig
    
    def plot_violin(
        self,
        column: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a violin plot for a numerical column.
        
        Args:
            column: Column name to visualize
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If column doesn't exist
            TypeError: If column is not numerical
        """
        self._validate_numeric(column)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        data = self.df[[column]].dropna()
        
        sns.violinplot(data=data, y=column, color='skyblue', ax=ax)
        
        ax.set_ylabel(column, fontsize=12, fontweight='bold')
        ax.set_title(title or f'Violin Plot of {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        logger.info(f"Violin plot created for '{column}'")
        return fig

    def plot_qq(
        self,
        column: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a Q-Q plot for a numerical column.

        Args:
            column: Column name to visualize
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.

        Returns:
            plt.Figure: Matplotlib figure object

        Raises:
            ValueError: If column doesn't exist
            TypeError: If column is not numerical
        """
        self._validate_numeric(column)

        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)

        data = self.df[column].dropna()
        stats.probplot(data, dist="norm", plot=ax)

        ax.set_title(title or f"Q-Q Plot of {column}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        logger.info(f"Q-Q plot created for '{column}'")
        return fig
    
    def plot_countplot(
        self,
        column: str,
        top_n: Optional[int] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a count plot for a categorical column.
        
        Args:
            column: Column name to visualize
            top_n: Show only top N categories. If None, shows all.
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If column doesn't exist
            TypeError: If column is too numeric to be categorical
        """
        self._validate_categorical(column)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        data = self.df[column].dropna()
        
        if top_n:
            value_counts = data.value_counts().head(top_n)
            data = data[data.isin(value_counts.index)]
        
        sns.countplot(y=data, order=data.value_counts().index, palette='Set2', ax=ax)
        
        ax.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax.set_ylabel(column, fontsize=12, fontweight='bold')
        ax.set_title(title or f'Count Plot of {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        logger.info(f"Count plot created for '{column}'")
        return fig
    
    def plot_pie_chart(
        self,
        column: str,
        top_n: int = 10,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a pie chart for a categorical column.
        
        Args:
            column: Column name to visualize
            top_n: Show only top N categories, combine rest as 'Others'
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If column doesn't exist
            TypeError: If column is too numeric to be categorical
        """
        self._validate_categorical(column)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        value_counts = self.df[column].value_counts()
        
        if len(value_counts) > top_n:
            top_values = value_counts.head(top_n)
            others_sum = value_counts.iloc[top_n:].sum()
            value_counts = pd.concat([top_values, pd.Series({'Others': others_sum})])
        
        colors = sns.color_palette('pastel', len(value_counts))
        
        ax.pie(
            value_counts.values,
            labels=value_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        
        ax.set_title(title or f'Distribution of {column}', fontsize=14, fontweight='bold')
        
        logger.info(f"Pie chart created for '{column}'")
        return fig
    
    # ==================== BIVARIATE ANALYSIS ====================
    
    def plot_scatter(
        self,
        x: str,
        y: str,
        hue: Optional[str] = None,
        size: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a scatter plot for two numerical columns.
        
        Args:
            x: Column name for x-axis
            y: Column name for y-axis
            hue: Optional column for color coding
            size: Optional column for point sizes
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If columns don't exist
            TypeError: If x or y are not numerical
        """
        self._validate_numeric(x)
        self._validate_numeric(y)
        if hue:
            self._validate_column(hue)
        if size:
            self._validate_numeric(size)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data
        cols_needed = [x, y]
        if hue:
            cols_needed.append(hue)
        if size:
            cols_needed.append(size)
        
        plot_data = self.df[cols_needed].dropna()
        
        # Create scatter plot
        if hue and size:
            scatter = ax.scatter(
                plot_data[x], plot_data[y],
                c=pd.Categorical(plot_data[hue]).codes if plot_data[hue].dtype == 'object' else plot_data[hue],
                s=plot_data[size] * 10,
                alpha=0.6, cmap='viridis', edgecolors='k', linewidth=0.5
            )
            plt.colorbar(scatter, ax=ax, label=hue)
        elif hue:
            for category in plot_data[hue].unique():
                mask = plot_data[hue] == category
                ax.scatter(plot_data.loc[mask, x], plot_data.loc[mask, y],
                          label=str(category), alpha=0.6, edgecolors='k', linewidth=0.5)
            ax.legend(title=hue)
        elif size:
            scatter = ax.scatter(plot_data[x], plot_data[y], s=plot_data[size] * 10,
                               alpha=0.6, c='steelblue', edgecolors='k', linewidth=0.5)
        else:
            ax.scatter(plot_data[x], plot_data[y], alpha=0.6, c='steelblue',
                      edgecolors='k', linewidth=0.5)
        
        ax.set_xlabel(x, fontsize=12, fontweight='bold')
        ax.set_ylabel(y, fontsize=12, fontweight='bold')
        ax.set_title(title or f'{y} vs {x}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        logger.info(f"Scatter plot created: {y} vs {x}")
        return fig
    
    def plot_regression(
        self,
        x: str,
        y: str,
        order: int = 1,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a scatter plot with regression line.
        
        Args:
            x: Column name for x-axis
            y: Column name for y-axis
            order: Polynomial order (1 for linear, 2 for quadratic, etc.)
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If columns don't exist
            TypeError: If x or y are not numerical
        """
        self._validate_numeric(x)
        self._validate_numeric(y)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_data = self.df[[x, y]].dropna()
        
        # Scatter plot
        ax.scatter(plot_data[x], plot_data[y], alpha=0.5, c='steelblue',
                  edgecolors='k', linewidth=0.5, label='Data')
        
        # Regression line
        sns.regplot(x=x, y=y, data=plot_data, ax=ax, scatter=False,
                   order=order, color='red', label=f'Regression (order={order})')
        
        # Calculate correlation
        corr = plot_data[x].corr(plot_data[y])
        ax.text(0.05, 0.95, f'Correlation: {corr:.3f}',
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(x, fontsize=12, fontweight='bold')
        ax.set_ylabel(y, fontsize=12, fontweight='bold')
        ax.set_title(title or f'{y} vs {x} (Regression)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        logger.info(f"Regression plot created: {y} vs {x}")
        return fig
    
    def plot_box_by_category(
        self,
        category: str,
        numeric: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create boxplots of a numeric variable grouped by categories.
        
        Args:
            category: Categorical column name
            numeric: Numerical column name
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If columns don't exist
            TypeError: If wrong data types
        """
        self._validate_categorical(category)
        self._validate_numeric(numeric)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_data = self.df[[category, numeric]].dropna()
        
        sns.boxplot(data=plot_data, x=category, y=numeric, palette='Set2', ax=ax)
        
        ax.set_xlabel(category, fontsize=12, fontweight='bold')
        ax.set_ylabel(numeric, fontsize=12, fontweight='bold')
        ax.set_title(title or f'{numeric} by {category}', fontsize=14, fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        logger.info(f"Box plot by category created: {numeric} by {category}")
        return fig
    
    def plot_violin_by_category(
        self,
        category: str,
        numeric: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create violin plots of a numeric variable grouped by categories.
        
        Args:
            category: Categorical column name
            numeric: Numerical column name
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If columns don't exist
            TypeError: If wrong data types
        """
        self._validate_categorical(category)
        self._validate_numeric(numeric)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_data = self.df[[category, numeric]].dropna()
        
        sns.violinplot(data=plot_data, x=category, y=numeric, palette='muted', ax=ax)
        
        ax.set_xlabel(category, fontsize=12, fontweight='bold')
        ax.set_ylabel(numeric, fontsize=12, fontweight='bold')
        ax.set_title(title or f'{numeric} by {category}', fontsize=14, fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        logger.info(f"Violin plot by category created: {numeric} by {category}")
        return fig
    
    def plot_grouped_countplot(
        self,
        cat1: str,
        cat2: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a grouped count plot for two categorical variables.
        
        Args:
            cat1: First categorical column name
            cat2: Second categorical column name (used for hue)
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If columns don't exist
            TypeError: If columns are not categorical
        """
        self._validate_categorical(cat1)
        self._validate_categorical(cat2)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_data = self.df[[cat1, cat2]].dropna()
        
        sns.countplot(data=plot_data, x=cat1, hue=cat2, palette='Set1', ax=ax)
        
        ax.set_xlabel(cat1, fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(title or f'{cat1} by {cat2}', fontsize=14, fontweight='bold')
        ax.legend(title=cat2)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        logger.info(f"Grouped count plot created: {cat1} by {cat2}")
        return fig
    
    def plot_crosstab_heatmap(
        self,
        cat1: str,
        cat2: str,
        normalize: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a heatmap of cross-tabulation for two categorical variables.
        
        Args:
            cat1: First categorical column name
            cat2: Second categorical column name
            normalize: Normalize by 'index', 'columns', or 'all'. None for counts.
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If columns don't exist
            TypeError: If columns are not categorical
        """
        self._validate_categorical(cat1)
        self._validate_categorical(cat2)
        
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create cross-tabulation
        crosstab = pd.crosstab(self.df[cat1], self.df[cat2], normalize=normalize)
        
        # Create heatmap
        sns.heatmap(crosstab, annot=True, fmt='.2%' if normalize else '.0f',
                   cmap='YlOrRd', cbar=True, ax=ax, linewidths=1)
        
        ax.set_xlabel(cat2, fontsize=12, fontweight='bold')
        ax.set_ylabel(cat1, fontsize=12, fontweight='bold')
        ax.set_title(title or f'Cross-tabulation: {cat1} vs {cat2}',
                    fontsize=14, fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
        plt.tight_layout()
        
        logger.info(f"Crosstab heatmap created: {cat1} vs {cat2}")
        return fig
    
    # ==================== MULTIVARIATE ANALYSIS ====================
    
    def plot_correlation_heatmap(
        self,
        columns: Optional[List[str]] = None,
        method: str = 'pearson',
        annot: bool = True,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a correlation heatmap for numerical columns.
        
        Args:
            columns: List of columns to include. If None, uses all numerical columns.
            method: Correlation method ('pearson', 'spearman', 'kendall')
            annot: Whether to annotate cells with correlation values
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, calculated based on number of columns.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If no numerical columns found or insufficient columns
        """
        # Get numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            raise ValueError("No numerical columns found for correlation heatmap")
        
        if len(numerical_cols) < 2:
            raise ValueError("Need at least 2 numerical columns for correlation")
        
        # Filter columns if specified
        if columns:
            missing_cols = set(columns) - set(numerical_cols)
            if missing_cols:
                raise ValueError(f"Columns not found or not numerical: {missing_cols}")
            cols_to_plot = columns
        else:
            cols_to_plot = numerical_cols
        
        # Calculate correlation matrix
        corr_matrix = self.df[cols_to_plot].corr(method=method)
        
        # Adjust figure size
        if figsize is None:
            size = max(8, len(cols_to_plot) * 0.8)
            figsize = (size, size)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            corr_matrix, cmap='coolwarm', center=0, annot=annot, fmt='.2f',
            square=True, linewidths=1, cbar_kws={'label': 'Correlation'},
            ax=ax, vmin=-1, vmax=1
        )
        
        ax.set_title(title or f'Correlation Matrix ({method.capitalize()})',
                    fontsize=14, fontweight='bold', pad=20)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
        plt.tight_layout()
        
        logger.info(f"Correlation heatmap created for {len(cols_to_plot)} columns")
        return fig
    
    def plot_pairplot(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        diag_kind: str = 'hist',
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a pairplot (scatterplot matrix) for numerical columns.
        
        Args:
            columns: List of columns to include. If None, uses all numerical columns.
            hue: Optional categorical column for color coding
            diag_kind: Type of plot for diagonal ('hist', 'kde')
            figsize: Not used (seaborn pairplot controls its own size)
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If no numerical columns found
        """
        # Get numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            raise ValueError("No numerical columns found for pairplot")
        
        if len(numerical_cols) < 2:
            raise ValueError("Need at least 2 numerical columns for pairplot")
        
        # Filter columns if specified
        if columns:
            missing_cols = set(columns) - set(numerical_cols)
            if missing_cols:
                raise ValueError(f"Columns not found or not numerical: {missing_cols}")
            cols_to_plot = columns
        else:
            # Limit to first 5 numerical columns to avoid overcrowding
            cols_to_plot = numerical_cols[:5]
        
        # Validate hue column if provided
        if hue:
            self._validate_column(hue)
            plot_cols = cols_to_plot + [hue]
        else:
            plot_cols = cols_to_plot
        
        # Create pairplot
        pairplot = sns.pairplot(
            self.df[plot_cols].dropna(),
            hue=hue,
            diag_kind=diag_kind,
            palette='husl',
            plot_kws={'alpha': 0.6, 'edgecolor': 'k', 'linewidth': 0.5},
            diag_kws={'alpha': 0.7}
        )
        
        pairplot.fig.suptitle('Pairplot of Numerical Features',
                             fontsize=16, fontweight='bold', y=1.02)
        
        logger.info(f"Pairplot created for {len(cols_to_plot)} columns")
        return pairplot.fig
    
    def plot_3d_scatter(
        self,
        x: str,
        y: str,
        z: str,
        color: Optional[str] = None,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create a 3D scatter plot for three numerical columns.
        
        Args:
            x: Column name for x-axis
            y: Column name for y-axis
            z: Column name for z-axis
            color: Optional column for color coding
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If columns don't exist
            TypeError: If x, y, or z are not numerical
        """
        self._validate_numeric(x)
        self._validate_numeric(y)
        self._validate_numeric(z)
        if color:
            self._validate_column(color)
        
        figsize = figsize or self.figsize
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare data
        cols_needed = [x, y, z]
        if color:
            cols_needed.append(color)
        
        plot_data = self.df[cols_needed].dropna()
        
        # Create 3D scatter
        if color:
            if pd.api.types.is_numeric_dtype(plot_data[color]):
                scatter = ax.scatter(
                    plot_data[x], plot_data[y], plot_data[z],
                    c=plot_data[color], cmap='viridis',
                    alpha=0.6, edgecolors='k', linewidth=0.5
                )
                fig.colorbar(scatter, ax=ax, label=color, shrink=0.5)
            else:
                for category in plot_data[color].unique():
                    mask = plot_data[color] == category
                    ax.scatter(
                        plot_data.loc[mask, x],
                        plot_data.loc[mask, y],
                        plot_data.loc[mask, z],
                        label=str(category), alpha=0.6,
                        edgecolors='k', linewidth=0.5
                    )
                ax.legend(title=color)
        else:
            ax.scatter(plot_data[x], plot_data[y], plot_data[z],
                      c='steelblue', alpha=0.6, edgecolors='k', linewidth=0.5)
        
        ax.set_xlabel(x, fontsize=11, fontweight='bold')
        ax.set_ylabel(y, fontsize=11, fontweight='bold')
        ax.set_zlabel(z, fontsize=11, fontweight='bold')
        ax.set_title(title or f'3D Scatter: {x}, {y}, {z}',
                    fontsize=14, fontweight='bold')
        
        logger.info(f"3D scatter plot created: {x}, {y}, {z}")
        return fig
    
    def plot_grouped_boxplot(
        self,
        category: str,
        numeric: str,
        hue: str,
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create grouped boxplots with a third categorical variable.
        
        Args:
            category: Primary categorical column (x-axis)
            numeric: Numerical column (y-axis)
            hue: Secondary categorical column (grouping)
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If columns don't exist
            TypeError: If wrong data types
        """
        self._validate_categorical(category)
        self._validate_numeric(numeric)
        self._validate_categorical(hue)
        
        figsize = figsize or (12, 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_data = self.df[[category, numeric, hue]].dropna()
        
        sns.boxplot(data=plot_data, x=category, y=numeric, hue=hue,
                   palette='Set2', ax=ax)
        
        ax.set_xlabel(category, fontsize=12, fontweight='bold')
        ax.set_ylabel(numeric, fontsize=12, fontweight='bold')
        ax.set_title(title or f'{numeric} by {category} and {hue}',
                    fontsize=14, fontweight='bold')
        ax.legend(title=hue)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        logger.info(f"Grouped boxplot created: {numeric} by {category} and {hue}")
        return fig
    
    def plot_multi_barplot(
        self,
        category: str,
        numeric: str,
        hue: str,
        aggregation: str = 'mean',
        title: Optional[str] = None,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        Create grouped bar plot showing aggregated values.
        
        Args:
            category: Primary categorical column (x-axis)
            numeric: Numerical column to aggregate (y-axis)
            hue: Secondary categorical column (grouping)
            aggregation: Aggregation method ('mean', 'sum', 'median', 'count')
            title: Plot title. If None, auto-generated.
            figsize: Figure size. If None, uses default.
        
        Returns:
            plt.Figure: Matplotlib figure object
            
        Raises:
            ValueError: If columns don't exist or invalid aggregation
            TypeError: If wrong data types
        """
        self._validate_categorical(category)
        self._validate_numeric(numeric)
        self._validate_categorical(hue)
        
        valid_aggs = ['mean', 'sum', 'median', 'count', 'min', 'max']
        if aggregation not in valid_aggs:
            raise ValueError(f"Invalid aggregation. Choose from {valid_aggs}")
        
        figsize = figsize or (12, 6)
        fig, ax = plt.subplots(figsize=figsize)
        
        plot_data = self.df[[category, numeric, hue]].dropna()
        
        sns.barplot(data=plot_data, x=category, y=numeric, hue=hue,
                   estimator=aggregation, palette='Set1', ax=ax, 
                   errorbar=None)
        
        ax.set_xlabel(category, fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{numeric} ({aggregation})', fontsize=12, fontweight='bold')
        ax.set_title(title or f'{numeric} {aggregation.capitalize()} by {category} and {hue}',
                    fontsize=14, fontweight='bold')
        ax.legend(title=hue)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        logger.info(f"Multi barplot created: {numeric} by {category} and {hue}")
        return fig
        logger.info(f"Multi barplot created: {numeric} by {category} and {hue}")
        return fig


# ==================== LEGACY METHODS (for backward compatibility) ====================

    def histogram(self, *args, **kwargs) -> plt.Figure:
        """Alias for plot_histogram for backward compatibility."""
        return self.plot_histogram(*args, **kwargs)
    
    def boxplot(self, columns: Optional[List[str]] = None, **kwargs) -> plt.Figure:
        """
        Legacy boxplot method showing multiple columns.
        For single column, use plot_boxplot instead.
        """
        if columns and len(columns) == 1:
            return self.plot_boxplot(columns[0], **kwargs)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            raise ValueError("No numerical columns found for boxplot")
        
        if columns:
            missing_cols = set(columns) - set(numerical_cols)
            if missing_cols:
                raise ValueError(f"Columns not found or not numerical: {missing_cols}")
            cols_to_plot = columns
        else:
            cols_to_plot = numerical_cols
        
        figsize = kwargs.get('figsize', self.figsize)
        fig, ax = plt.subplots(figsize=figsize)
        
        data_to_plot = [self.df[col].dropna() for col in cols_to_plot]
        
        bp = ax.boxplot(data_to_plot, labels=cols_to_plot, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(kwargs.get('title', 'Boxplot of Numerical Features'),
                    fontsize=14, fontweight='bold')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return fig
    
    def scatterplot(self, x_column: str, y_column: str, **kwargs) -> plt.Figure:
        """Alias for plot_scatter for backward compatibility."""
        return self.plot_scatter(x_column, y_column,
                                hue=kwargs.get('hue_column'),
                                size=kwargs.get('size_column'),
                                **{k: v for k, v in kwargs.items() 
                                   if k not in ['hue_column', 'size_column']})
    
    def correlation_heatmap(self, columns: Optional[List[str]] = None, **kwargs) -> plt.Figure:
        """Alias for plot_correlation_heatmap for backward compatibility."""
        return self.plot_correlation_heatmap(columns, **kwargs)
    
    def distribution_grid(self, columns: Optional[List[str]] = None, **kwargs) -> plt.Figure:
        """Legacy distribution grid method."""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numerical_cols:
            raise ValueError("No numerical columns found for distribution grid")
        
        if columns:
            missing_cols = set(columns) - set(numerical_cols)
            if missing_cols:
                raise ValueError(f"Columns not found or not numerical: {missing_cols}")
            cols_to_plot = columns
        else:
            cols_to_plot = numerical_cols
        
        n_cols = 3
        n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
        
        figsize = kwargs.get('figsize', (15, n_rows * 4))
        bins = kwargs.get('bins', 20)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(cols_to_plot):
            data = self.df[col].dropna()
            axes[idx].hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        for idx in range(len(cols_to_plot), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig


@dataclass
class ValidationMessage:
    level: str
    message: str


class BivariateAnalyzer:
    """
    Modular analyzer for bivariate analysis with clean separation of validation,
    statistics computation, and visualization.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        max_rows: int = 50000,
        sample: bool = True,
        random_state: int = 42
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df)}")
        if df.empty:
            raise ValueError("Cannot analyze an empty DataFrame")

        self.original_df = df
        self.sampled = False

        if sample and len(df) > max_rows:
            self.df = df.sample(max_rows, random_state=random_state)
            self.sampled = True
        else:
            self.df = df.copy()

    def detect_column_type(self, series: pd.Series) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "categorical"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        return "categorical"

    def detect_pair_type(self, col1: str, col2: str) -> str:
        type1 = self.detect_column_type(self.df[col1])
        type2 = self.detect_column_type(self.df[col2])
        if type1 == "numeric" and type2 == "numeric":
            return "numeric_numeric"
        if type1 == "categorical" and type2 == "categorical":
            return "categorical_categorical"
        return "categorical_numeric"

    def validate_selection(
        self,
        col1: str,
        col2: str,
        max_categories: int = 100,
        min_rows: int = 10
    ) -> List[ValidationMessage]:
        messages: List[ValidationMessage] = []

        if col1 not in self.df.columns or col2 not in self.df.columns:
            messages.append(ValidationMessage("error", "One or both selected columns are missing."))
            return messages

        if col1 == col2:
            messages.append(ValidationMessage("error", "Please select two different columns."))

        if self.df[col1].nunique(dropna=True) <= 1:
            messages.append(ValidationMessage("warning", f"{col1} has a single unique value."))

        if self.df[col2].nunique(dropna=True) <= 1:
            messages.append(ValidationMessage("warning", f"{col2} has a single unique value."))

        if self.detect_column_type(self.df[col1]) == "categorical":
            if self.df[col1].nunique(dropna=True) > max_categories:
                messages.append(ValidationMessage("error", f"{col1} has more than {max_categories} categories."))

        if self.detect_column_type(self.df[col2]) == "categorical":
            if self.df[col2].nunique(dropna=True) > max_categories:
                messages.append(ValidationMessage("error", f"{col2} has more than {max_categories} categories."))

        if len(self.df) < min_rows:
            messages.append(ValidationMessage("error", "Not enough rows for bivariate analysis."))

        return messages

    def _prepare_data(
        self,
        col1: str,
        col2: str,
        hue: Optional[str] = None,
        missing_strategy: str = "drop"
    ) -> pd.DataFrame:
        cols = [col1, col2]
        if hue:
            cols.append(hue)

        data = self.df[cols].copy()

        if missing_strategy == "treat_as_category":
            for col in [col1, col2]:
                if self.detect_column_type(data[col]) == "categorical":
                    data[col] = data[col].fillna("Missing")
        else:
            data = data.dropna()

        if hue and missing_strategy == "treat_as_category":
            if self.detect_column_type(data[hue]) == "categorical":
                data[hue] = data[hue].fillna("Missing")

        return data.dropna()

    def _strength_label(self, value: float) -> str:
        magnitude = abs(value)
        if magnitude < 0.3:
            return "weak"
        if magnitude < 0.6:
            return "moderate"
        return "strong"

    def compute_numeric_numeric_stats(
        self,
        col1: str,
        col2: str,
        missing_strategy: str = "drop"
    ) -> Dict[str, Any]:
        data = self._prepare_data(col1, col2, missing_strategy=missing_strategy)
        if len(data) < 3:
            return {"error": "Insufficient data after handling missing values."}

        pearson_r, pearson_p = stats.pearsonr(data[col1], data[col2])
        spearman_r, spearman_p = stats.spearmanr(data[col1], data[col2])

        return {
            "n": int(len(data)),
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "strength": self._strength_label(pearson_r)
        }

    def plot_numeric_numeric(
        self,
        col1: str,
        col2: str,
        plot_kind: str = "scatter",
        hue: Optional[str] = None,
        regression: bool = False,
        lowess: bool = False,
        hexbin: bool = False,
        kde_contour: bool = False,
        highlight_outliers: bool = False,
        palette: str = "husl",
        figsize: Tuple[int, int] = (10, 6),
        missing_strategy: str = "drop",
        annotation_text: Optional[str] = None,
        show_annotations: bool = False
    ) -> plt.Figure:
        data = self._prepare_data(col1, col2, hue=hue, missing_strategy=missing_strategy)
        if len(data) < 3:
            raise ValueError("Insufficient data for plotting")

        if plot_kind == "joint":
            grid = sns.jointplot(data=data, x=col1, y=col2, kind="scatter", height=figsize[0])
            grid.fig.suptitle(f"{col2} vs {col1}", y=1.02)
            return grid.fig

        fig, ax = plt.subplots(figsize=figsize)

        if plot_kind == "hexbin" or hexbin:
            ax.hexbin(data[col1], data[col2], gridsize=40, cmap="Blues", mincnt=1)
        else:
            if hue:
                sns.scatterplot(data=data, x=col1, y=col2, hue=hue, palette=palette, ax=ax, edgecolor='white', linewidth=0.4)
            else:
                ax.scatter(data[col1], data[col2], alpha=0.7, color="#1F3A8A", edgecolors='white', linewidth=0.4)

            if highlight_outliers:
                q1x, q3x = data[col1].quantile(0.25), data[col1].quantile(0.75)
                q1y, q3y = data[col2].quantile(0.25), data[col2].quantile(0.75)
                iqr_x, iqr_y = q3x - q1x, q3y - q1y
                mask = (
                    (data[col1] < q1x - 1.5 * iqr_x)
                    | (data[col1] > q3x + 1.5 * iqr_x)
                    | (data[col2] < q1y - 1.5 * iqr_y)
                    | (data[col2] > q3y + 1.5 * iqr_y)
                )
                ax.scatter(data.loc[mask, col1], data.loc[mask, col2], color="#EF4444", label="Outliers", alpha=0.7)

            if regression:
                sns.regplot(data=data, x=col1, y=col2, scatter=False, ax=ax, color="#0F766E")

            if lowess and STATSMODELS_AVAILABLE and sm_lowess is not None:
                lowess_fit = sm_lowess(data[col2], data[col1], frac=0.3)
                ax.plot(lowess_fit[:, 0], lowess_fit[:, 1], color="#F59E0B", linewidth=2, label="LOWESS")

        if kde_contour:
            sns.kdeplot(data=data, x=col1, y=col2, levels=6, color="#334155", ax=ax)

        if show_annotations and annotation_text:
            ax.text(
                0.01,
                0.99,
                annotation_text,
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )

        ax.set_xlabel(col1, fontsize=12, fontweight='bold')
        ax.set_ylabel(col2, fontsize=12, fontweight='bold')
        ax.set_title(f"{col2} vs {col1}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def compute_cat_num_stats(
        self,
        cat_col: str,
        num_col: str,
        missing_strategy: str = "drop"
    ) -> Dict[str, Any]:
        data = self._prepare_data(cat_col, num_col, missing_strategy=missing_strategy)
        groups = [grp[num_col].values for _, grp in data.groupby(cat_col)]
        if len(groups) < 2:
            return {"error": "Not enough groups for statistical tests."}

        anova_stat, anova_p = stats.f_oneway(*groups)
        kruskal_stat, kruskal_p = stats.kruskal(*groups)

        normal = False
        shapiro_pvals: List[float] = []
        if len(data) <= 5000:
            for group in groups:
                if len(group) >= 3:
                    stat, pval = stats.shapiro(group)
                    shapiro_pvals.append(pval)
            if shapiro_pvals:
                normal = all(pval > 0.05 for pval in shapiro_pvals)

        overall_mean = data[num_col].mean()
        ss_between = sum(len(grp) * (grp.mean() - overall_mean) ** 2 for _, grp in data.groupby(cat_col)[num_col])
        ss_total = ((data[num_col] - overall_mean) ** 2).sum()
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else np.nan

        return {
            "n": int(len(data)),
            "groups": int(len(groups)),
            "anova_stat": anova_stat,
            "anova_p": anova_p,
            "kruskal_stat": kruskal_stat,
            "kruskal_p": kruskal_p,
            "eta_sq": eta_sq,
            "normal": normal,
            "recommended_test": "ANOVA" if normal else "Kruskal-Wallis"
        }

    def plot_cat_num(
        self,
        cat_col: str,
        num_col: str,
        plot_kind: str = "box",
        top_n: Optional[int] = None,
        sort_by_mean: bool = True,
        show_ci: bool = True,
        palette: str = "Set2",
        figsize: Tuple[int, int] = (10, 6),
        missing_strategy: str = "drop",
        annotation_text: Optional[str] = None,
        show_annotations: bool = False
    ) -> plt.Figure:
        data = self._prepare_data(cat_col, num_col, missing_strategy=missing_strategy)
        if data.empty:
            raise ValueError("No data available after missing handling")

        if sort_by_mean:
            order = data.groupby(cat_col)[num_col].mean().sort_values(ascending=False).index
        else:
            order = data[cat_col].value_counts().index

        if top_n:
            order = order[:top_n]
            data = data[data[cat_col].isin(order)]

        fig, ax = plt.subplots(figsize=figsize)

        if plot_kind == "box":
            sns.boxplot(data=data, x=cat_col, y=num_col, order=order, palette=palette, ax=ax)
        elif plot_kind == "violin":
            sns.violinplot(data=data, x=cat_col, y=num_col, order=order, palette=palette, ax=ax)
        elif plot_kind == "swarm":
            sns.swarmplot(data=data, x=cat_col, y=num_col, order=order, palette=palette, size=4, ax=ax)
        elif plot_kind == "strip":
            sns.stripplot(data=data, x=cat_col, y=num_col, order=order, palette=palette, jitter=True, ax=ax)
        elif plot_kind == "mean_ci":
            sns.barplot(data=data, x=cat_col, y=num_col, order=order, palette=palette, ci=95 if show_ci else None, ax=ax)
        elif plot_kind == "ridgeline":
            g = sns.FacetGrid(data, row=cat_col, hue=cat_col, aspect=4, height=1.2, palette=palette, row_order=order)
            g.map(sns.kdeplot, num_col, fill=True, alpha=0.7)
            g.map(plt.axhline, y=0, lw=1, clip_on=False)
            g.fig.subplots_adjust(hspace=-0.3)
            g.set_titles("")
            g.set(yticks=[], ylabel="")
            g.fig.suptitle(f"{num_col} Distribution by {cat_col}", y=1.02)
            return g.fig
        else:
            sns.histplot(data=data, x=num_col, hue=cat_col, multiple="layer", stat="density", palette=palette, ax=ax)

        ax.set_title(f"{num_col} by {cat_col}", fontsize=14, fontweight='bold')
        if show_annotations and annotation_text:
            ax.text(
                0.01,
                0.99,
                annotation_text,
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return fig

    def compute_cat_cat_stats(
        self,
        col1: str,
        col2: str,
        normalize: str = "none",
        missing_strategy: str = "drop"
    ) -> Dict[str, Any]:
        data = self._prepare_data(col1, col2, missing_strategy=missing_strategy)
        crosstab = pd.crosstab(data[col1], data[col2])
        if crosstab.empty:
            return {"error": "No data available for categorical association."}

        chi2, p, _, expected = stats.chi2_contingency(crosstab)
        n = crosstab.to_numpy().sum()
        min_dim = min(crosstab.shape) - 1
        cramers_v = np.sqrt((chi2 / n) / min_dim) if min_dim > 0 else np.nan

        strength = self._strength_label(cramers_v) if not np.isnan(cramers_v) else "unknown"

        return {
            "n": int(n),
            "chi2": chi2,
            "p_value": p,
            "cramers_v": cramers_v,
            "strength": strength,
            "crosstab": crosstab
        }

    def plot_cat_cat(
        self,
        col1: str,
        col2: str,
        plot_kind: str = "heatmap",
        normalize: str = "none",
        palette: str = "Blues",
        figsize: Tuple[int, int] = (10, 6),
        missing_strategy: str = "drop",
        annotation_text: Optional[str] = None,
        show_annotations: bool = False
    ) -> plt.Figure:
        data = self._prepare_data(col1, col2, missing_strategy=missing_strategy)
        crosstab = pd.crosstab(data[col1], data[col2])
        if crosstab.empty:
            raise ValueError("No data available for plotting")

        if normalize == "row":
            plot_data = crosstab.div(crosstab.sum(axis=1), axis=0)
        elif normalize == "column":
            plot_data = crosstab.div(crosstab.sum(axis=0), axis=1)
        elif normalize == "overall":
            plot_data = crosstab / crosstab.values.sum()
        else:
            plot_data = crosstab

        if plot_kind == "mosaic" and STATSMODELS_AVAILABLE and sm_mosaic is not None:
            fig, ax = plt.subplots(figsize=figsize)
            sm_mosaic(data, [col1, col2], ax=ax, gap=0.01)
            ax.set_title(f"Mosaic Plot: {col1} vs {col2}")
            return fig

        fig, ax = plt.subplots(figsize=figsize)

        if plot_kind == "heatmap":
            sns.heatmap(plot_data, annot=True, fmt=".2f" if normalize != "none" else "d", cmap=palette, ax=ax)
        elif plot_kind == "stacked":
            plot_data.T.plot(kind="bar", stacked=True, ax=ax, colormap=palette)
        elif plot_kind == "grouped":
            plot_data.T.plot(kind="bar", ax=ax, colormap=palette)
        else:
            chi2, _, _, expected = stats.chi2_contingency(crosstab)
            residuals = (crosstab - expected) / np.sqrt(expected)
            sns.heatmap(residuals, center=0, cmap="coolwarm", ax=ax)

        ax.set_title(f"{col1} vs {col2}", fontsize=14, fontweight='bold')
        if show_annotations and annotation_text:
            ax.text(
                0.01,
                0.99,
                annotation_text,
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
        plt.tight_layout()
        return fig


def bivariate_analysis(df: pd.DataFrame, **kwargs: Any) -> BivariateAnalyzer:
    """Factory function for bivariate analysis."""
    return BivariateAnalyzer(df, **kwargs)


try:
    import squarify as _squarify
    SQUARIFY_AVAILABLE = True
except Exception:
    _squarify = None
    SQUARIFY_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose as _seasonal_decompose
    SEASONAL_AVAILABLE = True
except Exception:
    _seasonal_decompose = None
    SEASONAL_AVAILABLE = False


# ==================== UNIVARIATE ANALYSIS ENGINE ====================

@dataclass
class UnivariateValidation:
    level: str          # "error" | "warning" | "info"
    message: str


class UnivariateAnalyzer:
    """
    Fully modular univariate analysis engine.

    Separates column detection, statistical computation, and visualization.
    All plotting methods return plt.Figure; no Streamlit code inside.
    """

    MAX_ROWS_DEFAULT = 100_000
    RARE_THRESHOLD = 0.05
    HIGH_CARDINALITY_THRESHOLD = 50
    PALETTE = ["#1F3A8A", "#0F766E", "#334155", "#F59E0B", "#EF4444",
               "#10B981", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316"]

    def __init__(
        self,
        df: pd.DataFrame,
        max_rows: int = MAX_ROWS_DEFAULT,
        random_state: int = 42
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")
        if df.empty:
            raise ValueError("DataFrame is empty")

        self.original_df = df
        self.sampled = False

        if len(df) > max_rows:
            self.df = df.sample(max_rows, random_state=random_state)
            self.sampled = True
        else:
            self.df = df.copy()

    # ------------------------------------------------------------------
    # Column type detection
    # ------------------------------------------------------------------
    def detect_type(self, column: str) -> str:
        """Return one of: numeric | categorical | datetime | binary."""
        series = self.df[column]
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_bool_dtype(series):
            return "binary"
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique(dropna=True) == 2:
                return "binary"
            return "numeric"
        return "categorical"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, column: str) -> List[UnivariateValidation]:
        msgs: List[UnivariateValidation] = []
        if column not in self.df.columns:
            msgs.append(UnivariateValidation("error", f"Column '{column}' not found."))
            return msgs

        series = self.df[column]
        n = len(series)

        missing_pct = series.isna().mean() * 100
        if missing_pct == 100:
            msgs.append(UnivariateValidation("error", "Column is entirely missing."))
            return msgs
        if missing_pct > 50:
            msgs.append(UnivariateValidation("warning", f"{missing_pct:.1f}% values are missing."))
        elif missing_pct > 0:
            msgs.append(UnivariateValidation("info", f"{missing_pct:.1f}% missing values will be excluded."))

        if n < 10:
            msgs.append(UnivariateValidation("warning", "Very small dataset — statistics may be unreliable."))

        col_type = self.detect_type(column)
        if col_type == "numeric":
            if series.dropna().nunique() <= 1:
                msgs.append(UnivariateValidation("warning", "Column is constant — all values are identical."))
        elif col_type == "categorical":
            n_unique = series.nunique(dropna=True)
            if n_unique > 100:
                msgs.append(UnivariateValidation("error", f"Column has {n_unique} unique categories (>100). Use a filter."))
            elif n_unique > self.HIGH_CARDINALITY_THRESHOLD:
                msgs.append(UnivariateValidation("warning", f"High cardinality: {n_unique} unique categories."))

        if self.sampled:
            msgs.append(UnivariateValidation("info", f"Large dataset sampled to {len(self.df):,} rows for performance."))

        return msgs

    # ------------------------------------------------------------------
    # Numeric statistics
    # ------------------------------------------------------------------
    def compute_numeric_stats(self, column: str) -> Dict[str, Any]:
        series = self.df[column].dropna()
        if len(series) < 3:
            return {"error": "Not enough data to compute statistics."}

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())

        skew = series.skew()
        kurt = series.kurtosis()
        missing_pct = self.df[column].isna().mean() * 100

        # Normality tests
        shapiro_stat = shapiro_p = dagostino_stat = dagostino_p = np.nan
        if 3 <= len(series) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(series)
        if len(series) >= 8:
            dagostino_stat, dagostino_p = stats.normaltest(series)

        # Symmetry / tail interpretation
        if abs(skew) < 0.3:
            symmetry = "symmetric"
        elif abs(skew) < 1.0:
            symmetry = "moderately skewed"
        else:
            symmetry = "highly skewed"

        if kurt > 1:
            tails = "heavy tails (leptokurtic)"
        elif kurt < -1:
            tails = "light tails (platykurtic)"
        else:
            tails = "normal tails (mesokurtic)"

        return {
            "n": int(len(series)),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "mode": float(series.mode().iloc[0]) if not series.mode().empty else np.nan,
            "std": float(series.std()),
            "variance": float(series.var()),
            "skewness": float(skew),
            "kurtosis": float(kurt),
            "min": float(series.min()),
            "max": float(series.max()),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "outlier_count": outlier_count,
            "missing_pct": float(missing_pct),
            "shapiro_stat": float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
            "shapiro_p": float(shapiro_p) if not np.isnan(shapiro_p) else None,
            "dagostino_stat": float(dagostino_stat) if not np.isnan(dagostino_stat) else None,
            "dagostino_p": float(dagostino_p) if not np.isnan(dagostino_p) else None,
            "symmetry": symmetry,
            "tails": tails,
        }

    # ------------------------------------------------------------------
    # Categorical statistics
    # ------------------------------------------------------------------
    def compute_categorical_stats(self, column: str) -> Dict[str, Any]:
        series = self.df[column].dropna()
        value_counts = series.value_counts()
        total = len(series)

        probs = value_counts / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        missing_pct = self.df[column].isna().mean() * 100
        n_unique = int(series.nunique())

        if n_unique <= 5:
            cardinality = "low"
        elif n_unique <= self.HIGH_CARDINALITY_THRESHOLD:
            cardinality = "medium"
        else:
            cardinality = "high"

        rare = value_counts[value_counts / total < self.RARE_THRESHOLD]

        return {
            "n": int(total),
            "unique": n_unique,
            "mode": str(value_counts.index[0]),
            "mode_freq": int(value_counts.iloc[0]),
            "entropy": entropy,
            "missing_pct": float(missing_pct),
            "cardinality": cardinality,
            "rare_count": int(len(rare)),
            "value_counts": value_counts,
        }

    # ------------------------------------------------------------------
    # Datetime statistics
    # ------------------------------------------------------------------
    def compute_datetime_stats(self, column: str) -> Dict[str, Any]:
        series = pd.to_datetime(self.df[column], errors="coerce").dropna()
        span = series.max() - series.min()
        freq = pd.infer_freq(series.sort_values()) if len(series) > 2 else None
        return {
            "n": int(len(series)),
            "min": series.min(),
            "max": series.max(),
            "span_days": span.days,
            "inferred_freq": freq,
            "missing_pct": float(self.df[column].isna().mean() * 100),
        }

    # ------------------------------------------------------------------
    # Numeric plots
    # ------------------------------------------------------------------
    def _apply_common_style(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def plot_numeric(
        self,
        column: str,
        plot_kind: str = "histogram",
        bins: int = 30,
        kde_overlay: bool = False,
        fill_kde: bool = True,
        log_scale: bool = False,
        show_mean: bool = True,
        show_median: bool = True,
        highlight_outliers: bool = False,
        normal_overlay: bool = False,
        winsorize: bool = False,
        figsize: Tuple[int, int] = (10, 5),
    ) -> plt.Figure:
        raw = self.df[column].dropna()
        if winsorize:
            lower_p, upper_p = raw.quantile(0.05), raw.quantile(0.95)
            series = raw.clip(lower_p, upper_p)
        else:
            series = raw

        if log_scale:
            series = np.log1p(series[series > 0])
            col_label = f"log1p({column})"
        else:
            col_label = column

        if plot_kind == "histogram":
            fig, ax = plt.subplots(figsize=figsize)
            ax.hist(series, bins=bins, color=self.PALETTE[0], edgecolor='white', alpha=0.8, linewidth=0.5)
            if kde_overlay:
                ax2 = ax.twinx()
                sns.kdeplot(series, ax=ax2, color=self.PALETTE[1], linewidth=2)
                ax2.set_ylabel("Density", fontsize=10)
            if show_mean:
                ax.axvline(series.mean(), color=self.PALETTE[3], linestyle='--', linewidth=1.5, label=f"Mean={series.mean():.2f}")
            if show_median:
                ax.axvline(series.median(), color=self.PALETTE[2], linestyle=':', linewidth=1.5, label=f"Median={series.median():.2f}")
            if normal_overlay:
                mu, sigma = series.mean(), series.std()
                x = np.linspace(series.min(), series.max(), 200)
                from scipy.stats import norm as _norm
                y = _norm.pdf(x, mu, sigma) * len(series) * (series.max() - series.min()) / bins
                ax.plot(x, y, color=self.PALETTE[4], linewidth=2, label="Normal")
            if show_mean or show_median or normal_overlay:
                ax.legend(fontsize=9)
            if highlight_outliers:
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                mask = (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
                if mask.any():
                    ax.hist(series[mask], bins=bins, color=self.PALETTE[4], alpha=0.6, label="Outliers")
                    ax.legend(fontsize=9)
            self._apply_common_style(ax, f"Histogram of {col_label}", col_label, "Frequency")

        elif plot_kind == "kde":
            fig, ax = plt.subplots(figsize=figsize)
            sns.kdeplot(series, fill=fill_kde, color=self.PALETTE[0], ax=ax, linewidth=2)
            if show_mean:
                ax.axvline(series.mean(), color=self.PALETTE[3], linestyle='--', linewidth=1.5, label=f"Mean={series.mean():.2f}")
            if show_median:
                ax.axvline(series.median(), color=self.PALETTE[2], linestyle=':', linewidth=1.5, label=f"Median={series.median():.2f}")
            if show_mean or show_median:
                ax.legend(fontsize=9)
            self._apply_common_style(ax, f"KDE of {col_label}", col_label, "Density")

        elif plot_kind == "boxplot":
            fig, ax = plt.subplots(figsize=figsize)
            bp = ax.boxplot([series], labels=[col_label], patch_artist=True, vert=True)
            bp['boxes'][0].set_facecolor(self.PALETTE[0])
            bp['boxes'][0].set_alpha(0.7)
            for median in bp['medians']:
                median.set_color(self.PALETTE[3])
                median.set_linewidth(2)
            self._apply_common_style(ax, f"Boxplot of {col_label}", col_label, "Value")
            ax.grid(True, alpha=0.3, axis='y')

        elif plot_kind == "violin":
            fig, ax = plt.subplots(figsize=figsize)
            sns.violinplot(y=series, color=self.PALETTE[0], ax=ax, inner="box")
            self._apply_common_style(ax, f"Violin Plot of {col_label}", "", col_label)

        elif plot_kind == "rug":
            fig, ax = plt.subplots(figsize=figsize)
            sns.kdeplot(series, fill=True, color=self.PALETTE[0], ax=ax, linewidth=2, alpha=0.6)
            sns.rugplot(series, color=self.PALETTE[2], height=0.08, ax=ax)
            self._apply_common_style(ax, f"Rug Plot of {col_label}", col_label, "Density")

        elif plot_kind == "ecdf":
            fig, ax = plt.subplots(figsize=figsize)
            sorted_data = np.sort(series)
            ecdf_y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.step(sorted_data, ecdf_y, where='post', color=self.PALETTE[0], linewidth=2)
            ax.fill_between(sorted_data, ecdf_y, step='post', alpha=0.1, color=self.PALETTE[0])
            if show_mean:
                ax.axvline(series.mean(), color=self.PALETTE[3], linestyle='--', linewidth=1.5, label=f"Mean={series.mean():.2f}")
            if show_median:
                ax.axvline(series.median(), color=self.PALETTE[2], linestyle=':', linewidth=1.5, label=f"Median={series.median():.2f}")
            if show_mean or show_median:
                ax.legend(fontsize=9)
            self._apply_common_style(ax, f"ECDF of {col_label}", col_label, "Cumulative Probability")

        elif plot_kind == "qq":
            fig, ax = plt.subplots(figsize=figsize)
            stats.probplot(series, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot of {col_label}", fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

        elif plot_kind == "log_comparison":
            fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))
            axes[0].hist(series, bins=bins, color=self.PALETTE[0], edgecolor='white', alpha=0.8)
            axes[0].set_title(f"Original: {column}", fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            log_s = np.log1p(series[series > 0])
            axes[1].hist(log_s, bins=bins, color=self.PALETTE[1], edgecolor='white', alpha=0.8)
            axes[1].set_title(f"Log1p: {column}", fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

        else:
            raise ValueError(f"Unknown plot_kind '{plot_kind}'")

        plt.tight_layout()
        logger.info(f"Numeric plot '{plot_kind}' created for '{column}'")
        return fig

    # ------------------------------------------------------------------
    # Categorical plots
    # ------------------------------------------------------------------
    def plot_categorical(
        self,
        column: str,
        plot_kind: str = "bar",
        top_n: Optional[int] = None,
        group_rare: bool = False,
        sort_by_freq: bool = True,
        figsize: Tuple[int, int] = (10, 5),
    ) -> plt.Figure:
        series = self.df[column].dropna()
        value_counts = series.value_counts()
        total = len(series)

        if sort_by_freq:
            value_counts = value_counts.sort_values(ascending=False)
        else:
            value_counts = value_counts.sort_index()

        if top_n and top_n < len(value_counts):
            top = value_counts.head(top_n)
            if group_rare:
                others = value_counts.iloc[top_n:].sum()
                value_counts = pd.concat([top, pd.Series({"Others (grouped)": others})])
            else:
                value_counts = top

        labels = [str(lbl) for lbl in value_counts.index]
        values = value_counts.values
        pcts = values / total * 100

        if plot_kind == "bar":
            fig, ax = plt.subplots(figsize=figsize)
            bars = ax.bar(labels, values, color=self.PALETTE[:len(labels)], edgecolor='white')
            for bar, pct in zip(bars, pcts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + values.max() * 0.01,
                        f"{pct:.1f}%", ha='center', va='bottom', fontsize=8)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            self._apply_common_style(ax, f"Bar Chart: {column}", column, "Count")

        elif plot_kind == "hbar":
            fig, ax = plt.subplots(figsize=figsize)
            bars = ax.barh(labels[::-1], values[::-1], color=self.PALETTE[:len(labels)], edgecolor='white')
            for bar, pct in zip(bars, pcts[::-1]):
                ax.text(bar.get_width() + values.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.1f}%", ha='left', va='center', fontsize=8)
            self._apply_common_style(ax, f"Horizontal Bar: {column}", "Count", column)

        elif plot_kind == "pie":
            fig, ax = plt.subplots(figsize=figsize)
            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=self.PALETTE[:len(labels)], wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
            ax.set_title(f"Pie Chart: {column}", fontsize=14, fontweight='bold')

        elif plot_kind == "donut":
            fig, ax = plt.subplots(figsize=figsize)
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct='%1.1f%%', startangle=90,
                colors=self.PALETTE[:len(labels)], wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'width': 0.6}
            )
            ax.set_title(f"Donut Chart: {column}", fontsize=14, fontweight='bold')

        elif plot_kind == "pareto":
            fig, ax = plt.subplots(figsize=figsize)
            cumulative = np.cumsum(values) / total * 100
            ax.bar(labels, values, color=self.PALETTE[0], edgecolor='white', alpha=0.85)
            ax2 = ax.twinx()
            ax2.plot(labels, cumulative, color=self.PALETTE[4], marker='o', linewidth=2, markersize=5)
            ax2.axhline(80, color=self.PALETTE[3], linestyle='--', linewidth=1, label='80% line')
            ax2.set_ylabel("Cumulative %", fontsize=10, fontweight='bold')
            ax2.legend(fontsize=9)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            self._apply_common_style(ax, f"Pareto Chart: {column}", column, "Count")

        elif plot_kind == "treemap":
            if not SQUARIFY_AVAILABLE or _squarify is None:
                raise ImportError("squarify is not installed. Install it with: pip install squarify")
            fig, ax = plt.subplots(figsize=figsize)
            colors = self.PALETTE[:len(values)]
            treemap_labels = [f"{lbl}\n{pct:.1f}%" for lbl, pct in zip(labels, pcts)]
            _squarify.plot(sizes=values, label=treemap_labels, color=colors, alpha=0.85, ax=ax)
            ax.set_title(f"Treemap: {column}", fontsize=14, fontweight='bold')
            ax.axis('off')

        else:
            raise ValueError(f"Unknown plot_kind '{plot_kind}'")

        plt.tight_layout()
        logger.info(f"Categorical plot '{plot_kind}' created for '{column}'")
        return fig

    # ------------------------------------------------------------------
    # Datetime plots
    # ------------------------------------------------------------------
    def plot_datetime(
        self,
        column: str,
        plot_kind: str = "timeseries",
        figsize: Tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        series = pd.to_datetime(self.df[column], errors="coerce").dropna().sort_values()

        fig, ax = plt.subplots(figsize=figsize)

        if plot_kind == "timeseries":
            ax.plot(series, np.arange(len(series)), color=self.PALETTE[0], linewidth=1.5)
            self._apply_common_style(ax, f"Time Series: {column}", "Date", "Index")

        elif plot_kind == "monthly":
            monthly = series.dt.to_period("M").value_counts().sort_index()
            ax.bar([str(p) for p in monthly.index], monthly.values, color=self.PALETTE[0], edgecolor='white')
            ax.set_xticklabels([str(p) for p in monthly.index], rotation=45, ha='right', fontsize=8)
            self._apply_common_style(ax, f"Monthly Trend: {column}", "Month", "Count")

        elif plot_kind == "yearly":
            yearly = series.dt.year.value_counts().sort_index()
            ax.bar(yearly.index.astype(str), yearly.values, color=self.PALETTE[1], edgecolor='white')
            self._apply_common_style(ax, f"Yearly Trend: {column}", "Year", "Count")

        elif plot_kind == "dayofweek":
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow = series.dt.dayofweek.value_counts().reindex(range(7), fill_value=0)
            ax.bar(days, dow.values, color=self.PALETTE[2], edgecolor='white')
            self._apply_common_style(ax, f"Day-of-Week: {column}", "Day", "Count")

        elif plot_kind == "seasonal":
            if not SEASONAL_AVAILABLE or _seasonal_decompose is None:
                raise ImportError("statsmodels is required for seasonal decomposition.")
            ts = series.dt.to_period("M").value_counts().sort_index()
            ts_idx = ts.index.to_timestamp()
            ts_series = pd.Series(ts.values, index=ts_idx)
            if len(ts_series) < 4:
                raise ValueError("Not enough data points for seasonal decomposition.")
            result = _seasonal_decompose(ts_series, model='additive', period=min(12, len(ts_series) // 2))
            fig, axes = plt.subplots(4, 1, figsize=(figsize[0], figsize[1] * 2))
            result.observed.plot(ax=axes[0], color=self.PALETTE[0])
            axes[0].set_title("Observed", fontweight='bold')
            result.trend.plot(ax=axes[1], color=self.PALETTE[1])
            axes[1].set_title("Trend", fontweight='bold')
            result.seasonal.plot(ax=axes[2], color=self.PALETTE[2])
            axes[2].set_title("Seasonal", fontweight='bold')
            result.resid.plot(ax=axes[3], color=self.PALETTE[4])
            axes[3].set_title("Residual", fontweight='bold')
            plt.suptitle(f"Seasonal Decomposition: {column}", fontsize=14, fontweight='bold', y=1.01)

        else:
            raise ValueError(f"Unknown datetime plot_kind '{plot_kind}'")

        plt.tight_layout()
        return fig


def univariate_analysis(df: pd.DataFrame, **kwargs: Any) -> UnivariateAnalyzer:
    """Factory function for univariate analysis."""
    return UnivariateAnalyzer(df, **kwargs)


# ==================== OPTIONAL ADVANCED DEPS ====================

try:
    import umap as _umap_module  # type: ignore[import-untyped]
    UMAP_AVAILABLE = True
except Exception:
    _umap_module = None
    UMAP_AVAILABLE = False


# ==================== MULTIVARIATE ANALYSIS ENGINE ====================

class MultivariateAnalyzer:
    """
    Fully modular multivariate analysis engine.

    Sections:
    - Correlation (Pearson/Spearman/Kendall, VIF, multicollinearity)
    - Pairwise relationships (pairplot)
    - Dimensionality reduction (PCA, t-SNE, UMAP-optional)
    - Clustering (KMeans, Hierarchical, DBSCAN)
    - Feature interactions (3D scatter, bubble, parallel coords, radar, grouped heatmap)
    - Target-based analysis (correlation, RF importance, grouped stats)
    - Advanced statistics (covariance, partial correlation, condition number)

    All plot methods return plt.Figure. No Streamlit code inside.
    """

    MAX_ROWS_DEFAULT = 50_000
    PALETTE = [
        "#1F3A8A", "#0F766E", "#334155", "#F59E0B", "#EF4444",
        "#10B981", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316",
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        max_rows: int = MAX_ROWS_DEFAULT,
        random_state: int = 42,
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}")
        if df.empty:
            raise ValueError("DataFrame is empty")
        self.original_df = df
        self.sampled = False
        self.random_state = random_state
        if len(df) > max_rows:
            self.df = df.sample(max_rows, random_state=random_state)
            self.sampled = True
        else:
            self.df = df.copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_numeric(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        num_df = self.df.select_dtypes(include=[np.number])
        if columns:
            available = [c for c in columns if c in num_df.columns]
            num_df = num_df[available]
        # Drop duplicate column names to avoid reindex errors
        num_df = num_df.loc[:, ~num_df.columns.duplicated()]
        return num_df.dropna()

    def _scale(self, df: pd.DataFrame) -> np.ndarray:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler().fit_transform(df)

    def _apply_style(self, ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
        ax.set_title(title, fontsize=14, fontweight="bold")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # CORRELATION
    # ------------------------------------------------------------------
    def compute_correlation(
        self,
        method: str = "pearson",
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        num_df = self._get_numeric(columns)
        if num_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for correlation.")
        return num_df.corr(method=method)

    def strong_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.7
    ) -> pd.DataFrame:
        pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= threshold:
                    pairs.append({"Feature 1": cols[i], "Feature 2": cols[j], "Correlation": round(float(val), 4)})
        if not pairs:
            return pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation"])
        return pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False).reset_index(drop=True)

    def compute_vif(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            raise ImportError("statsmodels is required for VIF. Install with: pip install statsmodels")
        num_df = self._get_numeric(columns)
        if num_df.shape[1] < 2:
            raise ValueError("Need at least 2 columns for VIF.")
        X = num_df.values
        vif_rows = []
        for i in range(X.shape[1]):
            try:
                v = variance_inflation_factor(X, i)
            except Exception:
                v = float("inf")
            vif_rows.append({"Feature": num_df.columns[i], "VIF": round(float(v), 3)})
        return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)

    def plot_correlation_heatmap(
        self,
        method: str = "pearson",
        columns: Optional[List[str]] = None,
        mask_upper: bool = True,
        threshold: float = 0.0,
        annotate: bool = True,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        corr = self.compute_correlation(method=method, columns=columns)
        mask = np.triu(np.ones_like(corr.values, dtype=bool), k=1) if mask_upper else np.zeros_like(corr.values, dtype=bool)
        if threshold > 0:
            mask = mask | (np.abs(corr.values) < threshold)

        fig, ax = plt.subplots(figsize=figsize)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
            annot=annotate, fmt=".2f", square=True, linewidths=0.5,
            ax=ax, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8},
        )
        # Gold border on strong correlations
        strong = self.strong_correlations(corr, threshold=0.7)
        if not strong.empty:
            cols_list = list(corr.columns)
            for _, row in strong.iterrows():
                i = cols_list.index(row["Feature 1"])
                j = cols_list.index(row["Feature 2"])
                if not mask[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="gold", lw=2.5))
        ax.set_title(f"{method.capitalize()} Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def multicollinearity_report(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        vif = self.compute_vif(columns=columns)
        cond = self.compute_condition_number(columns=columns)
        corr = self.compute_correlation(columns=columns)
        strong = self.strong_correlations(corr, threshold=0.7)
        high_vif = vif[vif["VIF"] > 10]
        return {
            "vif": vif,
            "high_vif": high_vif,
            "condition_number": cond,
            "strong_correlations": strong,
        }

    # ------------------------------------------------------------------
    # PAIRWISE
    # ------------------------------------------------------------------
    def plot_pairplot(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        diag_kind: str = "kde",
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        num_df = self._get_numeric(columns)
        if num_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for pairplot.")
        if num_df.shape[1] > 5:
            num_df = num_df.iloc[:, :5]
        plot_df = num_df.copy()
        hue_col = None
        if hue and hue in self.df.columns:
            plot_df[hue] = self.df.loc[num_df.index, hue].astype(str)
            hue_col = hue
        n = plot_df.shape[1] - (1 if hue_col else 0)
        h = (figsize[0] if figsize else max(10, n * 3)) / n
        g = sns.pairplot(
            plot_df, hue=hue_col, diag_kind=diag_kind,
            plot_kws={"alpha": 0.45, "s": 15}, height=h,
        )
        g.figure.suptitle("Pairplot", y=1.02, fontsize=14, fontweight="bold")
        return g.figure

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------
    def compute_pca(
        self,
        columns: Optional[List[str]] = None,
        n_components: Optional[int] = None,
    ) -> Dict[str, Any]:
        from sklearn.decomposition import PCA
        num_df = self._get_numeric(columns)
        if num_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for PCA.")
        n = min(n_components or min(num_df.shape[1], 10), num_df.shape[1])
        X_scaled = self._scale(num_df)
        pca = PCA(n_components=n, random_state=self.random_state)
        components = pca.fit_transform(X_scaled)
        return {
            "components": components,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
            "n_components": n,
            "feature_names": list(num_df.columns),
            "loadings": pd.DataFrame(
                pca.components_.T,
                index=num_df.columns,
                columns=[f"PC{i+1}" for i in range(n)],
            ),
            "data_index": num_df.index,
        }

    def plot_pca_2d(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 7),
    ) -> plt.Figure:
        result = self.compute_pca(columns=columns, n_components=2)
        comp = result["components"]
        ev = result["explained_variance_ratio"]
        idx = result["data_index"]
        fig, ax = plt.subplots(figsize=figsize)
        if hue and hue in self.df.columns:
            hue_vals = self.df.loc[idx, hue].astype(str)
            for i, cat in enumerate(hue_vals.unique()):
                mask = (hue_vals == cat).values
                ax.scatter(comp[mask, 0], comp[mask, 1], label=cat, alpha=0.6, s=20,
                           color=self.PALETTE[i % len(self.PALETTE)])
            ax.legend(title=hue, fontsize=8, bbox_to_anchor=(1.02, 1))
        else:
            ax.scatter(comp[:, 0], comp[:, 1], alpha=0.6, s=20, color=self.PALETTE[0])
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=11, fontweight="bold")
        self._apply_style(ax, "PCA — 2D Projection")
        plt.tight_layout()
        return fig

    def plot_pca_3d(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        result = self.compute_pca(columns=columns, n_components=3)
        comp = result["components"]
        ev = result["explained_variance_ratio"]
        idx = result["data_index"]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        if hue and hue in self.df.columns:
            hue_vals = self.df.loc[idx, hue].astype(str)
            for i, cat in enumerate(hue_vals.unique()):
                mask = (hue_vals == cat).values
                ax.scatter(comp[mask, 0], comp[mask, 1], comp[mask, 2],
                           label=cat, alpha=0.6, s=15, color=self.PALETTE[i % len(self.PALETTE)])
            ax.legend(title=hue, fontsize=8)
        else:
            ax.scatter(comp[:, 0], comp[:, 1], comp[:, 2], alpha=0.6, s=15, color=self.PALETTE[0])
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=9)
        ax.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)", fontsize=9)
        ax.set_title("PCA — 3D Projection", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_scree(
        self,
        columns: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        result = self.compute_pca(columns=columns)
        ev = result["explained_variance_ratio"]
        cumvar = result["cumulative_variance"]
        labels = [f"PC{i+1}" for i in range(len(ev))]
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].bar(labels, ev * 100, color=self.PALETTE[0], edgecolor="white")
        self._apply_style(axes[0], "Scree Plot", "Principal Component", "Explained Variance %")
        axes[1].plot(labels, cumvar * 100, marker="o", color=self.PALETTE[1], linewidth=2)
        axes[1].axhline(80, linestyle="--", color=self.PALETTE[4], linewidth=1.2, label="80%")
        axes[1].axhline(95, linestyle="--", color=self.PALETTE[3], linewidth=1.2, label="95%")
        axes[1].legend(fontsize=9)
        self._apply_style(axes[1], "Cumulative Variance", "Principal Component", "Cumulative %")
        plt.suptitle("PCA Explained Variance", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------------
    def plot_tsne(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        figsize: Tuple[int, int] = (10, 7),
    ) -> plt.Figure:
        from sklearn.manifold import TSNE
        num_df = self._get_numeric(columns)
        if num_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for t-SNE.")
        if len(num_df) < 5:
            raise ValueError("Need at least 5 rows for t-SNE.")
        perplexity = min(float(perplexity), max(5.0, len(num_df) - 1))
        X_scaled = self._scale(num_df)
        # sklearn >= 1.2 renamed n_iter to max_iter; support both
        _tsne_kwargs: Dict[str, Any] = dict(
            n_components=2, perplexity=perplexity,
            learning_rate=learning_rate, random_state=self.random_state,
        )
        try:
            tsne = TSNE(max_iter=int(n_iter), **_tsne_kwargs)
        except TypeError:
            tsne = TSNE(n_iter=int(n_iter), **_tsne_kwargs)
        embedding = tsne.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=figsize)
        if hue and hue in self.df.columns:
            hue_vals = self.df.loc[num_df.index, hue].astype(str)
            for i, cat in enumerate(hue_vals.unique()):
                mask = (hue_vals == cat).values
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           label=cat, alpha=0.6, s=20, color=self.PALETTE[i % len(self.PALETTE)])
            ax.legend(title=hue, fontsize=8, bbox_to_anchor=(1.02, 1))
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20, color=self.PALETTE[0])
        self._apply_style(ax, f"t-SNE  (perplexity={perplexity:.0f}, lr={learning_rate:.0f})", "t-SNE 1", "t-SNE 2")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # UMAP (optional)
    # ------------------------------------------------------------------
    def plot_umap(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        figsize: Tuple[int, int] = (10, 7),
    ) -> plt.Figure:
        if not UMAP_AVAILABLE or _umap_module is None:
            raise ImportError("umap-learn is required. Install with: pip install umap-learn")
        num_df = self._get_numeric(columns)
        X_scaled = self._scale(num_df)
        reducer = _umap_module.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, random_state=self.random_state
        )
        embedding = reducer.fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=figsize)
        if hue and hue in self.df.columns:
            hue_vals = self.df.loc[num_df.index, hue].astype(str)
            for i, cat in enumerate(hue_vals.unique()):
                mask = (hue_vals == cat).values
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                           label=cat, alpha=0.6, s=20, color=self.PALETTE[i % len(self.PALETTE)])
            ax.legend(title=hue, fontsize=8, bbox_to_anchor=(1.02, 1))
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=20, color=self.PALETTE[0])
        self._apply_style(ax, f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})", "UMAP 1", "UMAP 2")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # CLUSTERING
    # ------------------------------------------------------------------
    def compute_elbow(
        self, columns: Optional[List[str]] = None, max_k: int = 10
    ) -> Dict[str, Any]:
        from sklearn.cluster import KMeans
        num_df = self._get_numeric(columns)
        X = self._scale(num_df)
        max_k = min(max_k, len(num_df) - 1, 15)
        ks = list(range(2, max_k + 1))
        inertias = []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=5)
            km.fit(X)
            inertias.append(km.inertia_)
        return {"k": ks, "inertia": inertias}

    def plot_elbow(
        self, columns: Optional[List[str]] = None, max_k: int = 10, figsize: Tuple[int, int] = (8, 5)
    ) -> plt.Figure:
        data = self.compute_elbow(columns=columns, max_k=max_k)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(data["k"], data["inertia"], marker="o", color=self.PALETTE[0], linewidth=2)
        self._apply_style(ax, "Elbow Method — KMeans Inertia", "Number of Clusters (k)", "Inertia")
        plt.tight_layout()
        return fig

    def fit_clusters(
        self,
        algorithm: str = "kmeans",
        columns: Optional[List[str]] = None,
        n_clusters: int = 3,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> Dict[str, Any]:
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        num_df = self._get_numeric(columns)
        if num_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for clustering.")
        X = self._scale(num_df)

        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X)
            scaler = StandardScaler().fit(num_df)
            centroids_df = pd.DataFrame(
                scaler.inverse_transform(model.cluster_centers_),
                columns=num_df.columns,
            )
            centroids_df.index.name = "Cluster"
        elif algorithm == "hierarchical":
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X)
            centroids_df = None
        elif algorithm == "dbscan":
            model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
            labels = model.fit_predict(X)
            centroids_df = None
        else:
            raise ValueError(f"Unknown algorithm: '{algorithm}'")

        valid_labels = [l for l in set(labels) if l != -1]
        sil_score = None
        if len(valid_labels) >= 2:
            try:
                sil_score = float(silhouette_score(X, labels))
            except Exception:
                pass

        cluster_sizes = (
            pd.Series(labels, name="Cluster")
            .value_counts().sort_index()
            .rename("Count").to_frame()
        )
        cluster_sizes.index.name = "Cluster"

        return {
            "labels": labels,
            "n_found": len(valid_labels),
            "silhouette": sil_score,
            "cluster_sizes": cluster_sizes,
            "centroids": centroids_df,
            "data": num_df,
            "columns": list(num_df.columns),
        }

    def plot_cluster_scatter(
        self,
        result: Dict[str, Any],
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 7),
    ) -> plt.Figure:
        data = result["data"]
        labels = result["labels"]
        cols = result["columns"]
        x_col = x_col or cols[0]
        y_col = y_col or (cols[1] if len(cols) > 1 else cols[0])
        fig, ax = plt.subplots(figsize=figsize)
        for lbl in sorted(set(labels)):
            mask = labels == lbl
            color = "#888888" if lbl == -1 else self.PALETTE[lbl % len(self.PALETTE)]
            ax.scatter(data[x_col].values[mask], data[y_col].values[mask],
                       label="Noise" if lbl == -1 else f"Cluster {lbl}",
                       alpha=0.6, s=20, color=color)
        ax.legend(fontsize=9)
        self._apply_style(ax, "Cluster Scatter Plot", x_col, y_col)
        plt.tight_layout()
        return fig

    def plot_cluster_scatter_3d(
        self,
        result: Dict[str, Any],
        x_col: Optional[str] = None,
        y_col: Optional[str] = None,
        z_col: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        data = result["data"]
        labels = result["labels"]
        cols = result["columns"]
        if len(cols) < 3:
            raise ValueError("Need at least 3 numeric columns for 3D cluster plot.")
        x_col = x_col or cols[0]
        y_col = y_col or cols[1]
        z_col = z_col or cols[2]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        for lbl in sorted(set(labels)):
            mask = labels == lbl
            color = "#888888" if lbl == -1 else self.PALETTE[lbl % len(self.PALETTE)]
            ax.scatter(data[x_col].values[mask], data[y_col].values[mask], data[z_col].values[mask],
                       label="Noise" if lbl == -1 else f"Cluster {lbl}",
                       alpha=0.6, s=15, color=color)
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
        ax.set_zlabel(z_col, fontsize=9)
        ax.set_title("Cluster 3D Scatter", fontsize=14, fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # FEATURE INTERACTIONS
    # ------------------------------------------------------------------
    def plot_3d_scatter(
        self,
        x_col: str,
        y_col: str,
        z_col: str,
        size_col: Optional[str] = None,
        color_col: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ) -> plt.Figure:
        sub = [x_col, y_col, z_col]
        if size_col:
            sub.append(size_col)
        df = self.df.dropna(subset=sub)
        sizes = None
        if size_col and size_col in df.columns:
            sv = df[size_col].astype(float)
            rng = sv.max() - sv.min()
            sizes = ((sv - sv.min()) / (rng + 1e-9) * 100 + 10).values
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        if color_col and color_col in df.columns:
            cats = df[color_col].astype(str)
            for i, cat in enumerate(cats.unique()):
                msk = (cats == cat).values
                sz = sizes[msk] if sizes is not None else 20
                ax.scatter(df[x_col].values[msk], df[y_col].values[msk], df[z_col].values[msk],
                           label=cat, color=self.PALETTE[i % len(self.PALETTE)], alpha=0.6, s=sz)
            ax.legend(title=color_col, fontsize=8)
        else:
            ax.scatter(df[x_col].values, df[y_col].values, df[z_col].values,
                       color=self.PALETTE[0], alpha=0.6, s=sizes if sizes is not None else 20)
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
        ax.set_zlabel(z_col, fontsize=9)
        size_label = f" · size={size_col}" if size_col else ""
        ax.set_title(f"3D Scatter: {x_col} / {y_col} / {z_col}{size_label}", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    def plot_parallel_coordinates(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        figsize: Tuple[int, int] = (13, 6),
    ) -> plt.Figure:
        from matplotlib.lines import Line2D
        num_df = self._get_numeric(columns)
        if num_df.shape[1] < 2:
            raise ValueError("Need at least 2 columns for parallel coordinates.")
        norm_df = (num_df - num_df.min()) / (num_df.max() - num_df.min() + 1e-9)
        col_names = list(norm_df.columns)
        hue_series = (
            self.df.loc[num_df.index, hue].astype(str)
            if hue and hue in self.df.columns
            else pd.Series(["All"] * len(num_df), index=num_df.index)
        )
        unique_cats = list(hue_series.unique())
        color_map = {c: self.PALETTE[i % len(self.PALETTE)] for i, c in enumerate(unique_cats)}
        fig, ax = plt.subplots(figsize=figsize)
        for row_idx in range(len(norm_df)):
            cat = hue_series.iloc[row_idx]
            ax.plot(range(len(col_names)), norm_df.values[row_idx], color=color_map[cat], alpha=0.25, linewidth=0.7)
        ax.set_xticks(range(len(col_names)))
        ax.set_xticklabels(col_names, rotation=35, ha="right", fontsize=9)
        handles = [Line2D([0], [0], color=color_map[c], linewidth=2, label=c) for c in unique_cats]
        ax.legend(handles=handles, title=hue or "Group", fontsize=8, bbox_to_anchor=(1.01, 1))
        self._apply_style(ax, "Parallel Coordinates (normalized)", "", "Normalized Value")
        plt.tight_layout()
        return fig

    def plot_radar(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8),
    ) -> plt.Figure:
        num_df = self._get_numeric(columns)
        col_names = list(num_df.columns)
        if len(col_names) < 3:
            raise ValueError("Need at least 3 columns for a radar chart.")
        norm_df = (num_df - num_df.min()) / (num_df.max() - num_df.min() + 1e-9)
        angles = np.linspace(0, 2 * np.pi, len(col_names), endpoint=False).tolist()
        angles_closed = angles + angles[:1]
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
        if hue and hue in self.df.columns:
            groups = self.df.loc[num_df.index, hue].astype(str)
            for i, cat in enumerate(list(groups.unique())[:6]):
                mask = (groups == cat).values
                vals = norm_df.values[mask].mean(axis=0).tolist()
                vals_closed = vals + vals[:1]
                ax.plot(angles_closed, vals_closed, color=self.PALETTE[i % len(self.PALETTE)], linewidth=2, label=cat)
                ax.fill(angles_closed, vals_closed, color=self.PALETTE[i % len(self.PALETTE)], alpha=0.1)
            ax.legend(fontsize=9, bbox_to_anchor=(1.25, 1))
        else:
            vals = norm_df.mean(axis=0).tolist()
            vals_closed = vals + vals[:1]
            ax.plot(angles_closed, vals_closed, color=self.PALETTE[0], linewidth=2)
            ax.fill(angles_closed, vals_closed, color=self.PALETTE[0], alpha=0.2)
        ax.set_xticks(angles)
        ax.set_xticklabels(col_names, fontsize=9)
        ax.set_title("Radar Chart (normalized group means)", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        return fig

    def plot_grouped_heatmap(
        self,
        group_col: str,
        columns: Optional[List[str]] = None,
        agg: str = "mean",
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        num_df = self._get_numeric(columns)
        if group_col not in self.df.columns:
            raise ValueError(f"Group column '{group_col}' not found.")
        combined = num_df.copy()
        combined[group_col] = self.df.loc[num_df.index, group_col]
        grouped = combined.groupby(group_col)[list(num_df.columns)].agg(agg)
        normed = (grouped - grouped.min()) / (grouped.max() - grouped.min() + 1e-9)
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(normed, annot=True, fmt=".2f", cmap="Blues", ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title(f"Grouped {agg.capitalize()} Heatmap by '{group_col}'", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # TARGET-BASED ANALYSIS
    # ------------------------------------------------------------------
    def compute_target_correlation(
        self,
        target: str,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
    ) -> pd.Series:
        num_df = self._get_numeric(columns)
        # Deduplicate columns to avoid reindex errors
        num_df = num_df.loc[:, ~num_df.columns.duplicated()]
        if target in num_df.columns:
            corr_series = num_df.corr(method=method)[target]
            corr_series = corr_series[corr_series.index != target]
            return corr_series.sort_values(key=abs, ascending=False)
        if target in self.df.columns:
            tgt = self.df.loc[num_df.index, target]
            if pd.api.types.is_numeric_dtype(tgt):
                corrs = num_df.apply(lambda col: col.corr(tgt))
                return corrs.sort_values(key=abs, ascending=False)
        raise ValueError(f"Target '{target}' must be a numeric column for correlation.")

    def compute_feature_importance(
        self,
        target: str,
        columns: Optional[List[str]] = None,
        task: str = "auto",
    ) -> pd.DataFrame:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        num_df = self._get_numeric(columns)
        if target not in self.df.columns:
            raise ValueError(f"Target '{target}' not found.")
        tgt = self.df.loc[num_df.index, target].dropna()
        X = num_df.loc[tgt.index]
        y = tgt
        if task == "auto":
            task = "classification" if (y.nunique() <= 20 and not pd.api.types.is_float_dtype(y)) else "regression"
        Model = RandomForestClassifier if task == "classification" else RandomForestRegressor
        model = Model(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        model.fit(X, y)
        return pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

    def plot_feature_importance(
        self,
        target: str,
        columns: Optional[List[str]] = None,
        task: str = "auto",
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        imp_df = self.compute_feature_importance(target=target, columns=columns, task=task)
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1],
                color=self.PALETTE[0], edgecolor="white")
        self._apply_style(ax, f"RF Feature Importance  →  {target}", "Importance", "")
        plt.tight_layout()
        return fig

    def compute_grouped_stats(
        self,
        target: str,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        num_df = self._get_numeric(columns)
        if target not in self.df.columns:
            raise ValueError(f"Target '{target}' not found.")
        combined = num_df.copy()
        combined[target] = self.df.loc[num_df.index, target]
        return combined.groupby(target).agg(["mean", "std", "count"]).round(3)

    # ------------------------------------------------------------------
    # ADVANCED STATISTICS
    # ------------------------------------------------------------------
    def compute_covariance(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        return self._get_numeric(columns).cov()

    def compute_condition_number(self, columns: Optional[List[str]] = None) -> float:
        num_df = self._get_numeric(columns)
        try:
            return float(np.linalg.cond(num_df.values))
        except Exception:
            return float("inf")

    def compute_partial_correlation(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        num_df = self._get_numeric(columns)
        if num_df.shape[1] < 3:
            raise ValueError("Need at least 3 columns for partial correlation.")
        try:
            prec = np.linalg.inv(num_df.cov().values)
            n = prec.shape[0]
            partial = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    partial[i, j] = -prec[i, j] / np.sqrt(np.abs(prec[i, i] * prec[j, j]) + 1e-12)
            np.fill_diagonal(partial, 1.0)
            return pd.DataFrame(partial, index=num_df.columns, columns=num_df.columns)
        except np.linalg.LinAlgError:
            raise ValueError("Singular covariance matrix — check for constant or duplicate columns.")


    def plot_target_correlation_bar(
        self,
        target: str,
        columns: Optional[List[str]] = None,
        method: str = "pearson",
        figsize: Tuple[int, int] = (10, 6),
    ) -> Tuple[plt.Figure, pd.DataFrame]:
        """Horizontal bar chart of feature correlations with a target column."""
        corrs = self.compute_target_correlation(target=target, columns=columns, method=method)
        corr_df = corrs.rename_axis("Feature").reset_index(name="Correlation")
        fig, ax = plt.subplots(figsize=(figsize[0], max(4, len(corr_df) * 0.4)))
        colors = [self.PALETTE[0] if v >= 0 else self.PALETTE[4] for v in corr_df["Correlation"]]
        ax.barh(corr_df["Feature"][::-1], corr_df["Correlation"][::-1], color=colors[::-1])
        ax.axvline(0, color="#333", linewidth=0.8)
        ax.set_title(f"{method.capitalize()} Correlation with '{target}'", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        return fig, corr_df

    def plot_covariance_heatmap(
        self,
        columns: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        cov_df = self.compute_covariance(columns=columns)
        fig, ax = plt.subplots(figsize=(min(figsize[0], len(cov_df) + 3), min(figsize[1], len(cov_df) + 2)))
        sns.heatmap(cov_df, annot=True, fmt=".2g", cmap="coolwarm", ax=ax,
                    linewidths=0.5, annot_kws={"size": 8})
        ax.set_title("Covariance Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig, cov_df

    def plot_partial_correlation_heatmap(
        self,
        columns: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        pcorr = self.compute_partial_correlation(columns=columns)
        fig, ax = plt.subplots(figsize=(min(figsize[0], len(pcorr) + 3), min(figsize[1], len(pcorr) + 2)))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(pcorr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1,
                    ax=ax, linewidths=0.5, annot_kws={"size": 8}, square=True)
        ax.set_title("Partial Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig


def multivariate_analysis(df: pd.DataFrame, **kwargs: Any) -> MultivariateAnalyzer:
    """Factory function for multivariate analysis."""
    return MultivariateAnalyzer(df, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    try:
        # Create sample data
        np.random.seed(42)
        sample_data = pd.DataFrame({
            'Age': np.random.normal(35, 10, 200),
            'Salary': np.random.normal(60000, 15000, 200),
            'Experience': np.random.normal(8, 3, 200),
            'Department': np.random.choice(['Sales', 'IT', 'HR', 'Marketing'], 200),
            'Satisfaction': np.random.randint(1, 6, 200),
            'Gender': np.random.choice(['Male', 'Female'], 200)
        })
        
        print("Sample Data Shape:", sample_data.shape)
        print("Columns:", list(sample_data.columns))
        print()
        
        # Initialize visualizer
        visualizer = EDAVisualizer(sample_data)
        
        print("=== UNIVARIATE ANALYSIS ===")
        print("Creating histogram...")
        fig1 = visualizer.plot_histogram('Age', bins=20)
        
        print("Creating KDE plot...")
        fig2 = visualizer.plot_kde('Salary')
        
        print("Creating boxplot...")
        fig3 = visualizer.plot_boxplot('Experience')
        
        print("Creating violin plot...")
        fig4 = visualizer.plot_violin('Age')
        
        print("Creating count plot...")
        fig5 = visualizer.plot_countplot('Department')
        
        print("Creating pie chart...")
        fig6 = visualizer.plot_pie_chart('Department')
        
        print("\n=== BIVARIATE ANALYSIS ===")
        print("Creating scatter plot...")
        fig7 = visualizer.plot_scatter('Experience', 'Salary', hue='Department')
        
        print("Creating regression plot...")
        fig8 = visualizer.plot_regression('Experience', 'Salary')
        
        print("Creating box by category...")
        fig9 = visualizer.plot_box_by_category('Department', 'Salary')
        
        print("Creating violin by category...")
        fig10 = visualizer.plot_violin_by_category('Department', 'Age')
        
        print("Creating grouped count plot...")
        fig11 = visualizer.plot_grouped_countplot('Department', 'Gender')
        
        print("Creating crosstab heatmap...")
        fig12 = visualizer.plot_crosstab_heatmap('Department', 'Gender')
        
        print("\n=== MULTIVARIATE ANALYSIS ===")
        print("Creating correlation heatmap...")
        fig13 = visualizer.plot_correlation_heatmap()
        
        print("Creating pairplot...")
        fig14 = visualizer.plot_pairplot(columns=['Age', 'Salary', 'Experience'])
        
        print("Creating 3D scatter...")
        fig15 = visualizer.plot_3d_scatter('Age', 'Salary', 'Experience', color='Department')
        
        print("Creating grouped boxplot...")
        fig16 = visualizer.plot_grouped_boxplot('Department', 'Salary', 'Gender')
        
        print("Creating multi barplot...")
        fig17 = visualizer.plot_multi_barplot('Department', 'Salary', 'Gender', aggregation='mean')
        
        print("\n✓ All visualizations created successfully!")
        print("Figure objects can be saved using: fig.savefig('filename.png')")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

