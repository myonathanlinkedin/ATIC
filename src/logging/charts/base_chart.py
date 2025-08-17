"""
Base chart utilities for benchmark visualization.
Provides common functionality for all chart modules.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseChart:
    """Base class providing common chart functionality"""
    
    def __init__(self, save_dir: str):
        """
        Initialize base chart utilities
        
        Args:
            save_dir: Directory to save generated charts
        """
        self.save_dir = save_dir
        self.charts_dir = os.path.join(save_dir, 'charts')
        os.makedirs(self.charts_dir, exist_ok=True)
        # Optional filename suffix per dataset (assigned externally by plotter)
        self.filename_suffix = ""
        # Internal flag to avoid duplicating dataset labels in titles
        self._title_decorated = False
        
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14

    def get_dataset_label(self) -> str:
        """Infer human-readable dataset label from filename suffix."""
        suffix = (self.filename_suffix or "").lower()
        if 'nmnist' in suffix:
            return 'N-MNIST'
        if 'shd' in suffix:
            return 'SHD'
        return ''

    def get_dynamic_x_range(
        self,
        data_length: int,
        max_steps: Optional[int] = None,
    ) -> range:
        """
        Generate dynamic x-axis range based on data length, optionally capped.
        
        Args:
            data_length: Length of the data array
            max_steps: Optional maximum steps. If None or <= 0, no cap is
                applied and the full length is used.
            
        Returns:
            Range object for x-axis
        """
        if max_steps is None or max_steps <= 0:
            return range(1, max(0, data_length) + 1)
        actual_length = min(data_length, max_steps)
        return range(1, max(0, actual_length) + 1)

    def apply_chart_improvements(
        self,
        ax,
        x_data,
        y_data_list,
        title: str,
        xlabel: str,
        ylabel: str,
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        caption: Optional[str] = None,
    ) -> None:
        """
        Apply standard chart improvements
        
        Args:
            ax: Matplotlib axis object
            x_data: X-axis data
            y_data_list: List of Y-axis data series
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            labels: List of labels for data series
            colors: List of colors for data series
            caption: Optional caption text
        """
        # Plot data series
        if colors is None:
            colors = ['red', 'blue', 'green', 'orange', 'purple']
        if labels is None:
            labels = [f'Series {i+1}' for i in range(len(y_data_list))]
        
        for i, y_data in enumerate(y_data_list):
            color = colors[i % len(colors)]
            label = labels[i] if i < len(labels) else f'Series {i+1}'
            ax.plot(x_data, y_data, 'o-', label=label, color=color,
                   linewidth=2, markersize=4)
        
        # Apply improvements
        dataset_label = self.get_dataset_label()
        decorated_title = f"{title} — {dataset_label}" if dataset_label else title
        ax.set_title(decorated_title, fontsize=14, fontweight='bold')
        self._title_decorated = True
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add caption if provided
        if caption:
            ax.text(
                0.02,
                0.98,
                caption,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            )
        
        # Auto-scale y-axis with margin
        if y_data_list:
            all_values = []
            for y_data in y_data_list:
                if y_data:
                    all_values.extend(y_data)
            if all_values:
                y_min, y_max = min(all_values), max(all_values)
                # Fix: Handle case where all values are identical
                if y_min == y_max:
                    margin = 0.1  # Use fixed margin for identical values
                    ax.set_ylim(max(0, y_min - margin), y_max + margin)
                else:
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim(max(0, y_min - margin), y_max + margin)

    def validate_data(self, data: Any, data_name: str = "data") -> bool:
        """
        Validate data before plotting
        
        Args:
            data: Data to validate
            data_name: Name of the data for error messages
            
        Returns:
            True if data is valid, False otherwise
        """
        if data is None:
            print(f"⚠️  Warning: {data_name} is None")
            return False
        
        if isinstance(data, list):
            if len(data) == 0:
                # Don't warn for empty lists - this is normal for some metrics
                # print(f"⚠️  Warning: {data_name} is empty list")
                return False
            # Allow data that contains only zeros or the same values
            # This is valid for metrics like GPU power and temperature when using fallback values
            if all(x is None for x in data):
                print(f"⚠️  Warning: {data_name} contains only None values")
                return False
            # Allow zeros and repeated values as they are valid for certain metrics
        elif isinstance(data, (int, float)):
            if data is None:
                print(f"⚠️  Warning: {data_name} is None")
                return False
        elif isinstance(data, dict):
            if len(data) == 0:
                print(f"⚠️  Warning: {data_name} is empty dict")
                return False
        
        return True
    
    def safe_extract_list_data(
        self,
        results: Dict[str, Any],
        key: str,
        default_length: int = None,  # Changed from hardcoded 20
    ) -> list:
        """
        Safely extract list data from results with validation
        
        Args:
            results: Results dictionary
            key: Key to extract
            default_length: Default length if data is missing (None = auto-detect)
            
        Returns:
            List of float values
        """
        try:
            data = results.get(key, [])

            # If data is missing or invalid, try to auto-detect epoch length
            if not self.validate_data(data, key):
                # Try to get epoch length from other available data
                auto_length = self._detect_epoch_length(results)
                if auto_length > 0:
                    return [0.0] * auto_length
                # Fallback to minimum length if no epoch info available
                return [0.0] * 1

            if isinstance(data, list):
                # Convert to float and handle None values, without padding
                float_data: List[float] = []
                for item in data:
                    if item is None:
                        float_data.append(0.0)
                    else:
                        try:
                            float_data.append(float(item))
                        except (ValueError, TypeError):
                            float_data.append(0.0)

                # Do not cap to a fixed length; always return full series
                # If list exists but empty, return zeros with detected epoch length
                if len(float_data) == 0:
                    auto_length = self._detect_epoch_length(results)
                    if auto_length > 0:
                        return [0.0] * auto_length
                    return [0.0] * 1
                return float_data

            # Single scalar value → return single-element series
            try:
                value = float(data)
                return [value]
            except (ValueError, TypeError):
                auto_length = self._detect_epoch_length(results)
                if auto_length > 0:
                    return [0.0] * auto_length
                return [0.0] * 1

        except Exception as e:
            print(f"⚠️  Warning: Error extracting {key}: {e}")
            return []

    def _detect_epoch_length(self, results: Dict[str, Any]) -> int:
        """
        Auto-detect epoch length from available data
        
        Args:
            results: Results dictionary
            
        Returns:
            Detected epoch length or 0 if not found
        """
        try:
            # Try to get epoch length from training data
            training_keys = ['training_accuracy', 'training_loss', 'validation_accuracy', 'validation_loss']
            for key in training_keys:
                if key in results:
                    data = results[key]
                    if isinstance(data, list) and len(data) > 0:
                        return len(data)
            
            # Try to get from SNN results
            if 'snn' in results and isinstance(results['snn'], dict):
                for key in training_keys:
                    if key in results['snn']:
                        data = results['snn'][key]
                        if isinstance(data, list) and len(data) > 0:
                            return len(data)
            
            # Try to get from ANN results
            if 'ann' in results and isinstance(results['ann'], dict):
                for key in training_keys:
                    if key in results['ann']:
                        data = results['ann'][key]
                        if isinstance(data, list) and len(data) > 0:
                            return len(data)
            
            # Try to get from comparison results
            if 'comparison' in results and isinstance(results['comparison'], dict):
                for key in training_keys:
                    if key in results['comparison']:
                        data = results['comparison'][key]
                        if isinstance(data, list) and len(data) > 0:
                            return len(data)
            
            return 0  # No epoch length detected
            
        except Exception:
            return 0

    def create_fallback_chart(self, filename: str, title: str):
        """
        Create a fallback chart when data is unavailable
        
        Args:
            filename: Output filename
            title: Chart title
        """
        try:
            # Auto-detect epoch length for fallback chart
            epoch_length = self._detect_epoch_length(self.results) if hasattr(self, 'results') else 10
            
            # Plot a zero-filled placeholder chart instead of "Data Not Available"
            _, ax = plt.subplots(figsize=(10, 6))
            x = list(range(1, epoch_length + 1))
            y = [0.0] * epoch_length
            ax.plot(x, y, 'o-', color='gray', linewidth=2, markersize=4)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            plt.savefig(
                os.path.join(
                    self.charts_dir,
                    f"{filename}{self.filename_suffix}.png",
                ),
                dpi=300,
                bbox_inches='tight',
            )
            plt.close()
            print(f"⚠️  Created fallback chart: {filename} with {epoch_length} epochs")
        except Exception as e:
            print(f"❌ Error creating fallback chart {filename}: {e}")

    def save_chart(self, filename: str):
        """
        Save the current chart
        
        Args:
            filename: Output filename
        """
        try:
            # If title wasn't decorated, add a small dataset tag on the figure
            if not self._title_decorated:
                label = self.get_dataset_label()
                if label:
                    fig = plt.gcf()
                    try:
                        fig.text(
                            0.99,
                            0.98,
                            label,
                            ha='right',
                            va='top',
                            fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        )
                    except Exception:
                        pass
            # Avoid using bbox_inches='tight' together with constrained_layout
            # to prevent excessive whitespace or oddly sized canvases
            plt.savefig(
                os.path.join(
                    self.charts_dir,
                    f"{filename}{self.filename_suffix}.png",
                ),
                dpi=300,
            )
            plt.close()
            print(f"✅ Generated {filename} chart")
            self._title_decorated = False
        except Exception as e:
            print(f"❌ Error saving chart {filename}: {e}")

    def safe_fill_between(self, ax, x_data, y1_data, y2_data, alpha=0.3,
                         **kwargs):
        """
        Safely fill between two data series
        
        Args:
            ax: Matplotlib axis object
            x_data: X-axis data
            y1_data: First Y-axis data series
            y2_data: Second Y-axis data series
            alpha: Transparency level
            **kwargs: Additional arguments for fill_between
        """
        try:
            if len(x_data) == len(y1_data) == len(y2_data):
                ax.fill_between(x_data, y1_data, y2_data, alpha=alpha,
                              **kwargs)
        except Exception as e:
            logger.warning("Error in safe_fill_between: %s", e)
