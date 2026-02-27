"""
Theme utilities for consistent visualization styling.
"""

from typing import Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)


# Theme definitions
THEME_LIBRARY = {
    "default": {
        "primary": "#1F3A8A",
        "secondary": "#334155",
        "accent": "#0F766E",
        "success": "#10B981",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "neutral": "#6B7280",
        "background": "#F8FAFC",
        "text": "#0F172A"
    },
    "dark": {
        "primary": "#3B82F6",
        "secondary": "#64748B",
        "accent": "#14B8A6",
        "success": "#22C55E",
        "warning": "#F59E0B",
        "error": "#EF4444",
        "neutral": "#9CA3AF",
        "background": "#1F2937",
        "text": "#F3F4F6"
    },
    "ocean": {
        "primary": "#0369A1",
        "secondary": "#0C4A6E",
        "accent": "#0EA5E9",
        "success": "#06B6D4",
        "warning": "#FBBF24",
        "error": "#F87171",
        "neutral": "#64748B",
        "background": "#F0F9FF",
        "text": "#082F49"
    },
    "forest": {
        "primary": "#15803D",
        "secondary": "#166534",
        "accent": "#16A34A",
        "success": "#22C55E",
        "warning": "#EAB308",
        "error": "#DC2626",
        "neutral": "#78350F",
        "background": "#F0FDF4",
        "text": "#1B4332"
    },
    "sunset": {
        "primary": "#D97706",
        "secondary": "#B45309",
        "accent": "#F59E0B",
        "success": "#84CC16",
        "warning": "#FC8181",
        "error": "#F87171",
        "neutral": "#78350F",
        "background": "#FFFBEB",
        "text": "#78350F"
    }
}


def get_theme_colors(theme_name: str = "default") -> Dict[str, str]:
    """
    Get color palette for a theme.
    
    Args:
        theme_name: Name of theme
        
    Returns:
        dict: Theme colors
    """
    if theme_name not in THEME_LIBRARY:
        logger.warning(f"Theme '{theme_name}' not found, using default")
        theme_name = "default"
    
    return THEME_LIBRARY[theme_name]


def apply_theme(theme_name: str = "default") -> Dict[str, str]:
    """
    Apply theme and return color configuration.
    
    Args:
        theme_name: Name of theme to apply
        
    Returns:
        dict: Applied theme colors
    """
    colors = get_theme_colors(theme_name)
    logger.info(f"Applied theme: {theme_name}")
    return colors


def list_available_themes() -> list:
    """
    Get list of available themes.
    
    Returns:
        list: Theme names
    """
    return list(THEME_LIBRARY.keys())


def create_custom_theme(
    name: str,
    colors: Dict[str, str]
) -> None:
    """
    Register a custom theme.
    
    Args:
        name: Theme name
        colors: Dictionary of colors
    """
    required_keys = {
        "primary", "secondary", "accent", "success",
        "warning", "error", "neutral", "background", "text"
    }
    
    if not required_keys.issubset(colors.keys()):
        missing = required_keys - colors.keys()
        logger.error(f"Missing required colors: {missing}")
        raise ValueError(f"Missing required colors: {missing}")
    
    THEME_LIBRARY[name] = colors
    logger.info(f"Registered custom theme: {name}")


if __name__ == "__main__":
    print("Available themes:")
    for theme in list_available_themes():
        print(f"  - {theme}")
    
    print("\nTheme colors:")
    colors = get_theme_colors()
    for name, color in colors.items():
        print(f"  {name}: {color}")
