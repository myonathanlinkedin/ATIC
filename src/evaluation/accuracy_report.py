import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.models.utils import set_seed

class AccuracyEvaluator:
    """Comprehensive accuracy evaluation for SNN vs ANN comparison"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        set_seed(42)
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """Evaluate model accuracy and performance metrics"""
        model.eval()
        model = model.to(self.device)
        
        all_predictions = []
        all_targets = []
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * total_correct / total_samples
        
        # Calculate additional metrics
        precision = self._calculate_precision(all_targets, all_predictions)
        recall = self._calculate_recall(all_targets, all_predictions)
        f1_score = self._calculate_f1_score(precision, recall)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_samples': total_samples,
            'correct_predictions': total_correct
        }
        
        return results
    
    def _calculate_precision(self, targets: List[int], predictions: List[int]) -> float:
        """Calculate precision"""
        if len(predictions) == 0:
            return 0.0
        
        correct = sum(1 for t, p in zip(targets, predictions) if t == p)
        return correct / len(predictions) if len(predictions) > 0 else 0.0
    
    def _calculate_recall(self, targets: List[int], predictions: List[int]) -> float:
        """Calculate recall"""
        if len(targets) == 0:
            return 0.0
        
        correct = sum(1 for t, p in zip(targets, predictions) if t == p)
        return correct / len(targets) if len(targets) > 0 else 0.0
    
    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        """Calculate F1 score"""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def generate_classification_report(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> str:
        """Generate detailed classification report"""
        model.eval()
        model = model.to(self.device)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        # Generate sklearn classification report
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(max(all_targets) + 1)]
        
        report = classification_report(
            all_targets,
            all_predictions,
            target_names=class_names,
            output_dict=True
        )
        
        return report
    
    def plot_confusion_matrix(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None,
        save_path: str = './results/charts/confusion_matrix.png'
    ):
        """Plot confusion matrix"""
        model.eval()
        model = model.to(self.device)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        test_loader: DataLoader,
        save_dir: str = './results'
    ) -> pd.DataFrame:
        """Compare multiple models and generate comparison report"""
        results = []
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, test_loader, model_name)
            metrics['model_name'] = model_name
            results.append(metrics)
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(os.path.join(save_dir, 'model_comparison.csv'), index=False)
        
        # Save detailed results as JSON
        detailed_results = {
            'comparison_timestamp': datetime.now().isoformat(),
            'models': results,
            'summary': {
                'best_accuracy': df['accuracy'].max(),
                'best_model': df.loc[df['accuracy'].idxmax(), 'model_name'],
                'average_accuracy': df['accuracy'].mean(),
                'std_accuracy': df['accuracy'].std()
            }
        }
        
        with open(os.path.join(save_dir, 'detailed_comparison.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        return df
    
    def generate_performance_summary(
        self,
        ann_results: Dict[str, float],
        snn_results: Dict[str, float],
        save_path: str = './results/performance_summary.json'
    ) -> Dict[str, Dict]:
        """Generate performance summary comparing ANN vs SNN"""
        summary = {
            'comparison_timestamp': datetime.now().isoformat(),
            'ann_results': ann_results,
            'snn_results': snn_results,
            'improvements': {
                'accuracy_improvement': snn_results['accuracy'] - ann_results['accuracy'],
                'precision_improvement': snn_results['precision'] - ann_results['precision'],
                'recall_improvement': snn_results['recall'] - ann_results['recall'],
                'f1_improvement': snn_results['f1_score'] - ann_results['f1_score']
            },
            'relative_improvements': {
                'accuracy_relative': (snn_results['accuracy'] - ann_results['accuracy']) / ann_results['accuracy'] * 100,
                'precision_relative': (snn_results['precision'] - ann_results['precision']) / ann_results['precision'] * 100,
                'recall_relative': (snn_results['recall'] - ann_results['recall']) / ann_results['recall'] * 100,
                'f1_relative': (snn_results['f1_score'] - ann_results['f1_score']) / ann_results['f1_score'] * 100
            }
        }
        
        # Save summary
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary 
