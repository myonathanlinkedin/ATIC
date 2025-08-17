"""
Brain Region Activation Chart Module
Data-driven visualization of brain region activation. Avoids any
handcrafted spatial templates; uses only what is present in results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List
import sys
import os

charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)
from base_chart import BaseChart  # noqa: E402  # type: ignore


class BrainRegionActivationChart(BaseChart):
    """Brain region activation patterns chart (fully data-driven)"""

    def _to_square_heatmap(self, values: List[float]) -> np.ndarray:
        """
        Convert a 1D sequence (e.g., per-epoch levels) into a proper
        heatmap that shows activation patterns clearly. For small datasets,
        use a simple row-based layout instead of forcing square/rectangular.
        """
        if not isinstance(values, list) or len(values) == 0:
            return np.zeros((1, 1), dtype=float)

        arr = np.array(
            [float(x) if x is not None else 0.0 for x in values],
            dtype=float,
        )
        
        # For small datasets (2-3 epochs), use simple row layout
        if len(arr) <= 3:
            # Simple row layout: each epoch is a row
            heat = arr.reshape(1, -1)  # 1 row, N columns
            return heat
        
        # For larger datasets, use rectangular grid
        total_epochs = len(arr)
        cols = max(3, int(np.ceil(np.sqrt(total_epochs))))
        rows = int(np.ceil(total_epochs / cols))
        heat = np.zeros((rows, cols), dtype=float)
        
        # Fill row by row to preserve epoch order
        for i, val in enumerate(arr):
            row = i // cols
            col = i % cols
            if row < rows and col < cols:
                heat[row, col] = val
        
        return heat

    def _generate_spatial_temporal_pattern(self, values: List[float], region_name: str) -> np.ndarray:
        """
        Generate natural spatial-temporal pattern from activation values.
        Creates a 10x10 grid that represents spatial-temporal activation patterns.
        """
        if not isinstance(values, list) or len(values) == 0:
            return np.zeros((10, 10), dtype=float)
        
        # Convert to numpy array
        arr = np.array([float(x) if x is not None else 0.0 for x in values], dtype=float)
        
        # Create 10x10 spatial-temporal grid
        grid = np.zeros((10, 10), dtype=float)
        
        if len(arr) > 0:
            # Normalize values to 0-1 range for better visualization
            if np.max(arr) > 0:
                normalized = arr / np.max(arr)
            else:
                normalized = arr
            
            # Generate spatial-temporal pattern based on region characteristics
            if region_name in ['V1', 'V1_primary']:
                # V1: Central focus pattern (primary visual cortex)
                center_x, center_y = 5, 5
                for i in range(10):
                    for j in range(10):
                        distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                        if distance <= 4:  # Central region
                            activation_idx = min(
                                int(distance * len(normalized) / 4), 
                                len(normalized) - 1
                            )
                            grid[i, j] = (normalized[activation_idx] 
                                         if activation_idx < len(normalized) 
                                         else 0.0)
                        else:
                            grid[i, j] = 0.0
                            
            elif region_name in ['V2', 'V2_secondary']:
                # V2: Horizontal band pattern (secondary visual processing)
                for i in range(10):
                    for j in range(10):
                        if 3 <= i <= 6:  # Horizontal band
                            activation_idx = min(j, len(normalized) - 1)
                            grid[i, j] = (normalized[activation_idx] 
                                         if activation_idx < len(normalized) 
                                         else 0.0)
                        else:
                            grid[i, j] = 0.0
                            
            elif region_name in ['V4', 'V4_color_form']:
                # V4: Enhanced central pattern (color and form processing)
                center_x, center_y = 5, 5
                for i in range(10):
                    for j in range(10):
                        distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                        if distance <= 5:  # Larger central region
                            activation_idx = min(
                                int(distance * len(normalized) / 5), 
                                len(normalized) - 1
                            )
                            grid[i, j] = (normalized[activation_idx] * 1.5 
                                         if activation_idx < len(normalized) 
                                         else 0.0)
                        else:
                            grid[i, j] = 0.0
                            
            elif region_name in ['IT', 'IT_object']:
                # IT: Distributed pattern (inferior temporal cortex)
                for i in range(10):
                    for j in range(10):
                        # Create two activation regions (upper-left and lower-right)
                        dist1 = np.sqrt((i - 2)**2 + (j - 2)**2)  # Upper-left
                        dist2 = np.sqrt((i - 7)**2 + (j - 7)**2)  # Lower-right
                        
                        if dist1 <= 2 or dist2 <= 2:
                            activation_idx = min(
                                int(min(dist1, dist2) * len(normalized) / 2), 
                                len(normalized) - 1
                            )
                            grid[i, j] = (normalized[activation_idx] 
                                         if activation_idx < len(normalized) 
                                         else 0.0)
                        else:
                            grid[i, j] = 0.0
        
        return grid

    def plot(self, results: Dict[str, Any]) -> None:
        try:
            snn_block = results.get('snn', {}) if isinstance(results, dict) else {}

            # 1) Preferred: explicit 2D activation maps per region from results
            #    Expected shape: dict { 'V1_primary': [[...], ...], ... }
            maps = snn_block.get('brain_activation_maps')
            if isinstance(maps, dict) and len(maps) > 0:
                regions = [
                    'V1_primary', 'V2_secondary', 'V4_color_form', 'IT_object'
                ]
                titles = [
                    'V1 Primary', 'V2 Secondary', 'V4 Color Form', 'IT Object'
                ]
                fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                                         constrained_layout=True)
                fig.suptitle('Brain Region Activation Patterns: SNN\n'
                             'Higher Activation Indicates Better Regional '
                             'Processing', fontsize=14, fontweight='bold')  # Reduced from 16 to 14

                for i, (region, title) in enumerate(zip(regions, titles)):
                    ax = axes[i // 2, i % 2]
                    data_2d = maps.get(region)
                    if (
                        isinstance(data_2d, list)
                        and len(data_2d) > 0
                        and isinstance(data_2d[0], list)
                    ):
                        mat = np.array(
                            [
                                [float(x) if x is not None else 0.0 for x in row]
                                for row in data_2d
                            ],
                            dtype=float,
                        )
                    else:
                        mat = np.zeros((1, 1), dtype=float)
                    im = ax.imshow(mat, cmap='hot', aspect='auto')
                    ax.set_title(title, fontsize=11, fontweight='bold')  # Reduced from 12 to 11
                    ax.set_xlabel('Spatial Position', fontsize=10)
                    ax.set_ylabel('Temporal Position', fontsize=10)
                    plt.colorbar(im, ax=ax, shrink=0.8)
                
                self.save_chart('18_brain_region_activation')
                return

            # 2) NEW: Show epoch progression for each brain region with better visualization
            #    This creates a comprehensive view of how each region evolves across epochs
            levels = snn_block.get('brain_activation_levels')
            if isinstance(levels, dict) and any(
                isinstance(v, list) and len(v) > 0 for v in levels.values()
            ):
                regions_map = {
                    'V1': levels.get('V1', []),
                    'V2': levels.get('V2', []),
                    'V4': levels.get('V4', []),
                    'IT': levels.get('IT', []),
                }
                regions = ['V1', 'V2', 'V4', 'IT']
                titles = [
                    'V1 Primary', 'V2 Secondary', 'V4 Color Form', 'IT Object'
                ]
                
                # Create epoch progression visualization
                total_epochs = len(next(iter(regions_map.values())))
                
                if total_epochs <= 10:
                    # For small number of epochs, show detailed progression
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                                           constrained_layout=True)
                    fig.suptitle('Brain Region Activation: Epoch Progression\n'
                               'Showing ALL Epochs with Detailed Patterns', 
                               fontsize=16, fontweight='bold')

                    for i, (region, title) in enumerate(zip(regions, titles)):
                        ax = axes[i // 2, i % 2]
                        values = regions_map.get(region, [])
                        
                        if values and len(values) > 0:
                            # Create epoch progression heatmap
                            epochs = range(1, len(values) + 1)
                            
                            # Reshape values into a 2D grid for better visualization
                            if len(values) <= 5:
                                # For very few epochs, use horizontal layout
                                grid = np.array(values).reshape(1, -1)
                                ax.imshow(grid, cmap='viridis', aspect='auto', 
                                         interpolation='nearest')
                                ax.set_xticks(range(len(values)))
                                ax.set_xticklabels([f'E{e}' for e in epochs])
                                ax.set_yticks([])
                                ax.set_ylabel('Activation')
                            else:
                                # For more epochs, create a proper grid
                                cols = min(8, len(values))
                                rows = int(np.ceil(len(values) / cols))
                                grid = np.zeros((rows, cols))
                                
                                for idx, val in enumerate(values):
                                    row = idx // cols
                                    col = idx % cols
                                    grid[row, col] = val
                                
                                im = ax.imshow(grid, cmap='viridis', aspect='auto', 
                                             interpolation='nearest')
                                ax.set_title(f'{title}\nMax: {max(values):.3f}', 
                                           fontsize=12, fontweight='bold')
                                
                                # Add epoch labels
                                ax.set_xticks(range(cols))
                                ax.set_xticklabels([f'E{i+1}' for i in range(cols)])
                                ax.set_yticks(range(rows))
                                ax.set_yticklabels([f'E{i+1}' for i in range(rows)])
                                
                                # Add colorbar
                                plt.colorbar(im, ax=ax, shrink=0.8)
                            
                            ax.set_xlabel('Epochs', fontsize=11)
                            ax.set_ylabel('Epochs', fontsize=11)
                            
                            # Add value annotations on grid
                            for row in range(grid.shape[0]):
                                for col in range(grid.shape[1]):
                                    val = grid[row, col]
                                    if val > 0:
                                        ax.text(col, row, f'{val:.2f}', 
                                               ha='center', va='center', 
                                               fontsize=8, fontweight='bold',
                                               color='white' if val > np.max(grid)/2 else 'black')
                        else:
                            ax.text(0.5, 0.5, 'No data',
                                   transform=ax.transAxes, ha='center', va='center',
                                   fontsize=14, bbox=dict(boxstyle='round,pad=0.5',
                                                        facecolor='lightyellow',
                                                        alpha=0.8))
                            ax.set_title(title, fontsize=12, fontweight='bold')

                else:
                    # For many epochs, show line chart progression
                    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                                           constrained_layout=True)
                    fig.suptitle('Brain Region Activation: Epoch Progression\n'
                               'Showing ALL Epochs with Line Charts', 
                               fontsize=16, fontweight='bold')

                    for i, (region, title) in enumerate(zip(regions, titles)):
                        ax = axes[i // 2, i % 2]
                        values = regions_map.get(region, [])
                        
                        if values and len(values) > 0:
                            epochs = range(1, len(values) + 1)
                            
                            # Plot epoch progression with distinct colors
                            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                                     '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
                            
                            # Use different line styles for better distinction
                            line_styles = ['-', '--', '-.', ':']
                            
                            # Plot with enhanced styling
                            ax.plot(epochs, values, 
                                   color=colors[i % len(colors)], 
                                   linestyle=line_styles[i % len(line_styles)],
                                   linewidth=3, marker='o', markersize=6,
                                   alpha=0.9, label=f'{title}')
                            
                            # Add value labels on points
                            for epoch, val in zip(epochs, values):
                                ax.annotate(f'{val:.3f}', 
                                           (epoch, val), 
                                           textcoords="offset points", 
                                           xytext=(0,10), 
                                           ha='center', fontsize=8)
                            
                            ax.set_title(f'{title}\nMax: {max(values):.3f}', 
                                       fontsize=12, fontweight='bold')
                            ax.set_xlabel('Epochs', fontsize=11)
                            ax.set_ylabel('Activation Level', fontsize=11)
                            ax.grid(True, alpha=0.3)
                            
                            # Set x-axis to show all epochs clearly
                            ax.set_xticks(epochs[::max(1, len(epochs)//10)])  # Show every nth epoch
                            ax.tick_params(axis='x', rotation=45)
                            
                            # Add trend line
                            if len(values) > 1:
                                z = np.polyfit(epochs, values, 1)
                                p = np.poly1d(z)
                                ax.plot(epochs, p(epochs), "--", alpha=0.5, 
                                       color='red', linewidth=1, label='Trend')
                                ax.legend(fontsize=9)
                        else:
                            ax.text(0.5, 0.5, 'No data',
                                   transform=ax.transAxes, ha='center', va='center',
                                   fontsize=14, bbox=dict(boxstyle='round,pad=0.5',
                                                        facecolor='lightyellow',
                                                        alpha=0.8))
                            ax.set_title(title, fontsize=12, fontweight='bold')

                self.save_chart('18_brain_region_activation')
                return

            # 3) Legacy: simple scalar per region -> show bar chart
            #    (still data-driven)
            legacy = snn_block.get('brain_activation')
            if isinstance(legacy, dict) and len(legacy) > 0:
                regions = [
                    'V1_primary', 'V2_secondary', 'V4_color_form', 'IT_object'
                ]
                titles = [
                    'V1 Primary', 'V2 Secondary', 'V4 Color Form', 'IT Object'
                ]
                values = [
                    float(legacy.get(r, {}).get('level', 0.0)) for r in regions
                ]

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(
                    titles,
                    values,
                    color=['#d62728', '#1f77b4', '#2ca02c', '#9467bd'],
                    alpha=0.85,
                )
                ax.set_title(
                    'Brain Region Activation Levels: SNN',
                    fontsize=14,
                    fontweight='bold',
                )
                ax.set_ylabel('Activation Level (relative)')
                ax.grid(True, axis='y', alpha=0.3)
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.3f}",
                        ha='center',
                        va='bottom',
                        fontsize=9,
                    )

                self.save_chart('18_brain_region_activation')
                return

            # If nothing available, create a neutral fallback
            self.create_fallback_chart('18_brain_region_activation', 'Brain Region Activation')
        except Exception as e:
            print(f"❌ Error in brain region activation chart: {e}")
            self.create_fallback_chart('18_brain_region_activation', 'Brain Region Activation')
