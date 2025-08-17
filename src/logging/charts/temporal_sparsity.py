"""
Temporal Sparsity Chart Module
Generates temporal sparsity 3D surface plot for SNN models.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any
import sys
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
charts_dir = os.path.dirname(os.path.abspath(__file__))
if charts_dir not in sys.path:
    sys.path.insert(0, charts_dir)
from base_chart import BaseChart
class TemporalSparsityChart(BaseChart):
    """Temporal sparsity 3D surface chart"""
    def plot(self, results: Dict[str, Any]) -> None:
        try:
            # Try to get temporal sparsity data from SNN results
            sparsity_data = None
            epoch_sparsity_data = None
            
            # Check if results has 'snn' key
            if 'snn' in results and isinstance(results['snn'], dict):
                sparsity_data = results['snn'].get('temporal_sparsity', None)
                epoch_sparsity_data = results['snn'].get('epoch_temporal_sparsity', None)
            else:
                # Fallback: check if temporal_sparsity is directly in results
                sparsity_data = results.get('temporal_sparsity', None)
                epoch_sparsity_data = results.get('epoch_temporal_sparsity', None)
            
            # PRIORITY 1: Enhanced 3D surface from enhanced matrix data
            if sparsity_data is not None and isinstance(sparsity_data, (list, np.ndarray)) and len(sparsity_data) > 0:
                print(f"✅ Using enhanced temporal sparsity matrix for 3D surface")
                self._plot_enhanced_3d_surface(sparsity_data, epoch_sparsity_data)
                return
            
            # PRIORITY 2: Epoch-specific temporal sparsity (2D progression)
            if epoch_sparsity_data is not None and isinstance(epoch_sparsity_data, (list, np.ndarray)) and len(epoch_sparsity_data) > 0:
                print(f"⚠️  Using epoch-specific temporal sparsity data: {len(epoch_sparsity_data)} epochs")
                self._plot_epoch_temporal_sparsity(epoch_sparsity_data)
                return
            
            # PRIORITY 3: No data available
            print(f"⚠️  Warning: No valid temporal sparsity data found. Data type: {type(sparsity_data)}")
            self.create_fallback_chart('21_temporal_sparsity', 'Temporal Sparsity 3D Surface')
            
        except Exception as e:
            print(f"❌ Error in temporal sparsity chart: {e}")
            self.create_fallback_chart('21_temporal_sparsity', 'Temporal Sparsity 3D Surface')
    
    def _plot_enhanced_3d_surface(self, sparsity_data, epoch_data=None) -> None:
        """Plot enhanced 3D surface with meaningful variations"""
        try:
            # Convert to numpy array
            if isinstance(sparsity_data, list):
                Z = np.array(sparsity_data)
            else:
                Z = sparsity_data
            
            # Ensure it's a 2D array for 3D surface
            if Z.ndim == 1:
                # Reshape 1D array to 2D (assuming square matrix)
                size = int(np.sqrt(len(Z)))
                if size * size == len(Z):
                    Z = Z.reshape(size, size)
                else:
                    # Pad to make it square
                    size = int(np.ceil(np.sqrt(len(Z))))
                    padded = np.zeros(size * size)
                    padded[:len(Z)] = Z
                    Z = padded.reshape(size, size)
            
            # Ensure minimum size for visualization
            if Z.shape[0] < 3 or Z.shape[1] < 3:
                # Pad to minimum size
                min_size = max(3, Z.shape[0], Z.shape[1])
                padded = np.zeros((min_size, min_size))
                padded[:Z.shape[0], :Z.shape[1]] = Z
                Z = padded
            
            # Create meshgrid for 3D surface
            x = np.linspace(0, 10, Z.shape[1])
            y = np.linspace(0, 10, Z.shape[0])
            X, Y = np.meshgrid(x, y)
            
            # Show only 3D surface (removed 2D chart)
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                                 linewidth=0.5, edgecolor='black')
            ax.set_xlabel('Temporal Position', fontsize=12)
            ax.set_ylabel('Spatial Position', fontsize=12)
            ax.set_zlabel('Sparsity Level', fontsize=12)
            ax.set_title('Enhanced Temporal Sparsity 3D Surface\nMulti-Scale Analysis with Real Variations', 
                        fontsize=14, fontweight='bold')  # Reduced from 16 to 14
            
            # Add colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
            
            plt.tight_layout(pad=3.0)
            self.save_chart('21_temporal_sparsity')
            
            # Print matrix statistics for verification
            print(f"📊 Enhanced 3D Matrix Statistics:")
            print(f"   Shape: {Z.shape}")
            print(f"   Min/Max: {Z.min():.6f}/{Z.max():.6f}")
            print(f"   Mean: {Z.mean():.6f}")
            print(f"   Std: {Z.std():.6f}")
            print(f"   Range: {Z.max() - Z.min():.6f}")
            
        except Exception as e:
            print(f"❌ Error in enhanced 3D surface plotting: {e}")
            self.create_fallback_chart('21_temporal_sparsity', 'Enhanced Temporal Sparsity 3D Surface')
    
    def _plot_epoch_temporal_sparsity(self, epoch_data: list) -> None:
        """Plot temporal sparsity progression over epochs"""
        try:
            # Convert to numpy array
            if isinstance(epoch_data, list):
                Z = np.array(epoch_data)
            else:
                Z = epoch_data
            
            # Create epoch progression chart
            epochs = range(1, len(Z) + 1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Chart 1: Line plot showing sparsity progression
            ax1.plot(epochs, Z, 'b-o', linewidth=2, markersize=6)
            ax1.set_xlabel('Training Epoch', fontsize=12)
            ax1.set_ylabel('Temporal Sparsity', fontsize=12)
            ax1.set_title('Temporal Sparsity Progression Over Training', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(1, len(Z))
            
            # Chart 2: 3D surface showing sparsity matrix (if we have enough data)
            if len(Z) >= 10:
                # Create a 2D matrix by repeating the 1D data
                matrix_size = min(10, len(Z))
                Z_matrix = np.zeros((matrix_size, matrix_size))
                for i in range(matrix_size):
                    for j in range(matrix_size):
                        Z_matrix[i, j] = Z[min(i, len(Z)-1)]
                
                x = np.linspace(0, 10, matrix_size)
                y = np.linspace(0, 10, matrix_size)
                X, Y = np.meshgrid(x, y)
                
                ax2 = fig.add_subplot(122, projection='3d')
                surf = ax2.plot_surface(X, Y, Z_matrix, cmap='viridis', alpha=0.8)
                ax2.set_xlabel('Temporal Position', fontsize=12)
                ax2.set_ylabel('Spatial Position', fontsize=12)
                ax2.set_zlabel('Sparsity Level', fontsize=12)
                ax2.set_title('Temporal Sparsity 3D Surface (Epoch-Based)', fontsize=14, fontweight='bold')
            else:
                # If not enough epochs, show histogram
                ax2.hist(Z, bins=min(10, len(Z)), alpha=0.7, color='skyblue', edgecolor='black')
                ax2.set_xlabel('Temporal Sparsity', fontsize=12)
                ax2.set_ylabel('Frequency', fontsize=12)
                ax2.set_title('Temporal Sparsity Distribution', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout(pad=3.0)
            self.save_chart('21_temporal_sparsity')
            
        except Exception as e:
            print(f"❌ Error in epoch temporal sparsity plotting: {e}")
            self.create_fallback_chart('21_temporal_sparsity', 'Temporal Sparsity Epoch Progression')
    
    def _plot_matrix_temporal_sparsity(self, sparsity_data) -> None:
        """Plot traditional 3D surface from matrix data (fallback)"""
        try:
            # Convert to numpy array if it's a list
            if isinstance(sparsity_data, list):
                Z = np.array(sparsity_data)
            else:
                Z = sparsity_data
            
            # Ensure it's a 2D array for 3D surface
            if Z.ndim == 1:
                # Reshape 1D array to 2D (assuming square matrix)
                size = int(np.sqrt(len(Z)))
                if size * size == len(Z):
                    Z = Z.reshape(size, size)
                else:
                    # Pad to make it square
                    size = int(np.ceil(np.sqrt(len(Z))))
                    padded = np.zeros(size * size)
                    padded[:len(Z)] = Z
                    Z = padded.reshape(size, size)
            
            # Ensure minimum size for visualization
            if Z.shape[0] < 3 or Z.shape[1] < 3:
                # Pad to minimum size
                min_size = max(3, Z.shape[0], Z.shape[1])
                padded = np.zeros((min_size, min_size))
                padded[:Z.shape[0], :Z.shape[1]] = Z
                Z = padded
            
            # Create meshgrid for 3D surface
            x = np.linspace(0, 10, Z.shape[1])
            y = np.linspace(0, 10, Z.shape[0])
            X, Y = np.meshgrid(x, y)
            
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
            ax.set_xlabel('Temporal Position', fontsize=12)
            ax.set_ylabel('Spatial Position', fontsize=12)
            ax.set_zlabel('Sparsity Level', fontsize=12)
            ax.set_title('Temporal Sparsity 3D Surface: SNN\nLower Sparsity Indicates More Dense Neural Activity', fontsize=16, fontweight='bold')
            plt.tight_layout(pad=3.0)
            self.save_chart('21_temporal_sparsity')
            
        except Exception as e:
            print(f"❌ Error in matrix temporal sparsity plotting: {e}")
            self.create_fallback_chart('21_temporal_sparsity', 'Temporal Sparsity 3D Surface')
