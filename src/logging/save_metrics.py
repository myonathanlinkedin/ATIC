import json
import pandas as pd
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

class MetricsLogger:
    """
    Comprehensive metrics logging and saving
    
    PURPOSE:
    - Save and manage comprehensive benchmark metrics
    - Generate summary reports and analysis
    - Create LaTeX tables for research papers
    - Provide detailed logging and documentation
    
    FUNCTIONALITY:
    - Training metrics saving (accuracy, loss, time)
    - Benchmark results logging
    - Model comparison analysis
    - Summary report generation
    - LaTeX table creation for papers
    
    OUTPUT FORMATS:
    - JSON files for structured data storage
    - CSV files for spreadsheet analysis
    - LaTeX tables for research papers
    - Summary reports with key findings
    
    METRICS TRACKED:
    - Performance metrics (accuracy, loss)
    - Resource metrics (time, memory, energy)
    - Model complexity (parameters, layers)
    - Comparison analysis (improvements, reductions)
    """
    
    def __init__(self, save_dir: str = './results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'charts'), exist_ok=True)
    
    def save_training_metrics(
        self,
        model_name: str,
        dataset_name: str,
        metrics: Dict[str, List],
        save_path: Optional[str] = None
    ):
        """Save training metrics to JSON and CSV"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, 'logs', f'{model_name}_{dataset_name}_{timestamp}.json')
        
        # Add metadata
        metrics_data = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        # Save as JSON
        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_path = save_path.replace('.json', '.csv')
        self._save_metrics_to_csv(metrics, csv_path)
        
        print(f"Training metrics saved to: {save_path}")
        return save_path
    
    def _save_metrics_to_csv(self, metrics: Dict[str, List], csv_path: str):
        """Convert metrics to CSV format"""
        # Create DataFrame from metrics
        df_data = {}
        for key, values in metrics.items():
            if isinstance(values, list):
                df_data[key] = values
            else:
                df_data[key] = [values]
        
        # Pad shorter lists with NaN
        max_length = max(len(v) for v in df_data.values() if isinstance(v, list))
        for key, values in df_data.items():
            if isinstance(values, list) and len(values) < max_length:
                df_data[key].extend([np.nan] * (max_length - len(values)))
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
    
    def save_benchmark_results(
        self,
        results: Dict[str, Any],
        dataset_name: str,
        save_path: Optional[str] = None
    ):
        """Save complete benchmark results"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'{dataset_name}_benchmark_{timestamp}.json')
        
        # Add metadata
        results_data = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Benchmark results saved to: {save_path}")
        return save_path
    
    def save_model_comparison(
        self,
        comparison_data: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Save model comparison results"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'model_comparison_{timestamp}.json')
        
        comparison_data['timestamp'] = datetime.now().isoformat()
        
        with open(save_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"Model comparison saved to: {save_path}")
        return save_path
    
    def create_summary_report(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """Create comprehensive summary report"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'summary_report_{timestamp}.json')
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'datasets_tested': list(results.keys()),
            'total_experiments': len(results),
            'key_findings': {},
            'performance_summary': {},
            'energy_summary': {}
        }
        
        for dataset, result in results.items():
            # Handle pending datasets
            if result.get('pending', False):
                summary['performance_summary'][dataset] = {
                    'ann_accuracy': None,
                    'snn_accuracy': None,
                    'accuracy_improvement': None,
                    'accuracy_improvement_percent': None
                }
                summary['energy_summary'][dataset] = {
                    'ann_energy_joules': None,
                    'snn_energy_joules': None,
                    'energy_reduction_joules': None,
                    'energy_reduction_percent': None
                }
                continue
            
            # Performance summary
            ann_acc = result.get('ann_accuracy', {}).get('accuracy', 0)
            snn_acc = result.get('snn_accuracy', {}).get('accuracy', 0)
            
            summary['performance_summary'][dataset] = {
                'ann_accuracy': ann_acc,
                'snn_accuracy': snn_acc,
                'accuracy_improvement': snn_acc - ann_acc,
                'accuracy_improvement_percent': ((snn_acc - ann_acc) / ann_acc * 100) if ann_acc > 0 else 0
            }
            
            # Energy summary
            if 'energy_comparison' in result:
                energy_comp = result['energy_comparison']
                
                def safe_extract_energy(energy_data):
                    if isinstance(energy_data, dict):
                        if 'total_energy_j' in energy_data:
                            return float(energy_data['total_energy_j'])
                        if 'total_energy_joules' in energy_data:
                            return float(energy_data['total_energy_joules'])
                        if 'energy' in energy_data:
                            return float(energy_data['energy'])
                        if 'value' in energy_data:
                            return float(energy_data['value'])
                        return 0.0
                    return float(energy_data) if isinstance(energy_data, (int, float)) else 0.0
                
                ann_energy = safe_extract_energy(energy_comp.get('ann_inference', {}).get('total_energy_joules', 0))
                snn_energy = safe_extract_energy(energy_comp.get('snn_inference', {}).get('total_energy_joules', 0))
                
                summary['energy_summary'][dataset] = {
                    'ann_energy_joules': ann_energy,
                    'snn_energy_joules': snn_energy,
                    'energy_reduction_joules': ann_energy - snn_energy,
                    'energy_reduction_percent': ((ann_energy - snn_energy) / ann_energy * 100) if ann_energy > 0 else 0
                }
        
        # Aggregate findings using only available datasets
        valid_acc = [v['accuracy_improvement_percent'] for v in summary['performance_summary'].values() if v['accuracy_improvement_percent'] is not None]
        avg_accuracy_improvement = float(np.mean(valid_acc)) if valid_acc else 0.0
        
        valid_energy = [v['energy_reduction_percent'] for v in summary['energy_summary'].values() if v['energy_reduction_percent'] is not None]
        avg_energy_reduction = float(np.mean(valid_energy)) if valid_energy else 0.0
        
        best_performing_dataset = None
        most_energy_efficient_dataset = None
        if valid_acc:
            best_performing_dataset = max(
                [k for k, v in summary['performance_summary'].items() if v['accuracy_improvement_percent'] is not None],
                key=lambda x: summary['performance_summary'][x]['accuracy_improvement_percent']
            )
        if valid_energy:
            most_energy_efficient_dataset = max(
                [k for k, v in summary['energy_summary'].items() if v['energy_reduction_percent'] is not None],
                key=lambda x: summary['energy_summary'][x]['energy_reduction_percent']
            )
        
        summary['key_findings'] = {
            'average_accuracy_improvement_percent': avg_accuracy_improvement,
            'average_energy_reduction_percent': avg_energy_reduction,
            'best_performing_dataset': best_performing_dataset,
            'most_energy_efficient_dataset': most_energy_efficient_dataset
        }
        
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary report saved to: {save_path}")
        return save_path
    
    def save_etad_analysis(
        self,
        decay_lambdas: List[float],
        accuracies: List[float],
        energy_consumptions: List[float],
        save_path: Optional[str] = None
    ):
        """Save ETAD parameter analysis results"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'etad_analysis_{timestamp}.json')
        
        analysis_data = {
            'timestamp': datetime.now().isoformat(),
            'decay_lambdas': decay_lambdas,
            'accuracies': accuracies,
            'energy_consumptions': energy_consumptions,
            'optimal_decay_lambda': decay_lambdas[np.argmax(accuracies)],
            'best_accuracy': max(accuracies),
            'lowest_energy': min(energy_consumptions)
        }
        
        with open(save_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"ETAD analysis saved to: {save_path}")
        return save_path
    
    def create_latex_table(
        self,
        results: Dict[str, Dict],
        save_path: Optional[str] = None
    ):
        """Create LaTeX table for paper inclusion"""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'benchmark_table_{timestamp}.tex')
        
        latex_content = []
        latex_content.append(r"\begin{table}[htbp]")
        latex_content.append(r"\centering")
        latex_content.append(r"\caption{SNN vs ANN Benchmark Results}")
        latex_content.append(r"\label{tab:benchmark_results}")
        latex_content.append(r"\begin{tabular}{lcccc}")
        latex_content.append(r"\toprule")
        latex_content.append(r"Dataset & Model & Accuracy (\%) & Energy (J) & Time (s) \\")
        latex_content.append(r"\midrule")
        
        def fmt(value, pending=False):
            if pending:
                return r"\text{N/A}"
            return f"{value:.2f}"
        
        for dataset, result in results.items():
            is_pending = result.get('pending', False)
            if is_pending:
                latex_content.append(f"{dataset} & ANN & {fmt(0, True)} & {fmt(0, True)} & {fmt(0, True)} \\\\")
                latex_content.append(f" & SNN+ETAD & {fmt(0, True)} & {fmt(0, True)} & {fmt(0, True)} \\\\")
                latex_content.append(r"\addlinespace")
                continue
            
            def safe_extract_latex_value(data_dict, key, default=0):
                value = data_dict.get(key, default)
                if isinstance(value, dict):
                    if 'value' in value:
                        return float(value['value'])
                    if 'total' in value:
                        return float(value['total'])
                    if 'total_energy_j' in value:
                        return float(value['total_energy_j'])
                    if 'total_energy_joules' in value:
                        return float(value['total_energy_joules'])
                    return float(default)
                return float(value) if isinstance(value, (int, float)) else float(default)
            
            ann_acc = safe_extract_latex_value(result.get('ann_accuracy', {}), 'accuracy', 0)
            ann_energy_data = result.get('energy_comparison', {}).get('ann_inference', {}).get('total_energy_joules', {})
            ann_energy = ann_energy_data.get('total_energy_j', 0.0) if isinstance(ann_energy_data, dict) else float(ann_energy_data or 0.0)
            ann_time = safe_extract_latex_value(result.get('energy_comparison', {}).get('ann_inference', {}), 'total_inference_time_s', 0)
            latex_content.append(f"{dataset} & ANN & {ann_acc:.2f} & {ann_energy:.2f} & {ann_time:.3f} \\\\")
            
            snn_acc = safe_extract_latex_value(result.get('snn_accuracy', {}), 'accuracy', 0)
            snn_energy_data = result.get('energy_comparison', {}).get('snn_inference', {}).get('total_energy_joules', {})
            snn_energy = snn_energy_data.get('total_energy_j', 0.0) if isinstance(snn_energy_data, dict) else float(snn_energy_data or 0.0)
            snn_time = safe_extract_latex_value(result.get('energy_comparison', {}).get('snn_inference', {}), 'total_inference_time_s', 0)
            latex_content.append(f" & SNN+ETAD & {snn_acc:.2f} & {snn_energy:.2f} & {snn_time:.3f} \\\\")
            latex_content.append(r"\addlinespace")
        
        latex_content.append(r"\bottomrule")
        latex_content.append(r"\end{tabular}")
        latex_content.append(r"\end{table}")
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(latex_content))
        
        print(f"LaTeX table saved to: {save_path}")
        return save_path

    def create_appendix_charts(self, results: Dict[str, Dict]):
        """Create minimal appendix charts per dataset (final accuracy, inference time, energy per sample)."""
        import matplotlib.pyplot as plt
        import numpy as np
        charts_path = os.path.join(self.save_dir, 'charts')
        os.makedirs(charts_path, exist_ok=True)
        
        for dataset, result in results.items():
            if result.get('pending', False):
                continue
            try:
                # Extract values with fallbacks
                ann_acc = float(result.get('ann_accuracy', {}).get('accuracy', 0.0))
                snn_acc = float(result.get('snn_accuracy', {}).get('accuracy', 0.0))
                ann_energy = result.get('energy_comparison', {}).get('ann_inference', {}).get('total_energy_joules', 0.0)
                if isinstance(ann_energy, dict):
                    ann_energy = float(ann_energy.get('total_energy_j', 0.0))
                else:
                    ann_energy = float(ann_energy or 0.0)
                snn_energy = result.get('energy_comparison', {}).get('snn_inference', {}).get('total_energy_joules', 0.0)
                if isinstance(snn_energy, dict):
                    snn_energy = float(snn_energy.get('total_energy_j', 0.0))
                else:
                    snn_energy = float(snn_energy or 0.0)
                ann_time = float(result.get('energy_comparison', {}).get('ann_inference', {}).get('total_inference_time_s', 0.0))
                snn_time = float(result.get('energy_comparison', {}).get('snn_inference', {}).get('total_inference_time_s', 0.0))
                # Normalize accuracy if >1
                if ann_acc > 1.0:
                    ann_acc /= 100.0
                if snn_acc > 1.0:
                    snn_acc /= 100.0
                # 1) Final Accuracy
                fig, ax = plt.subplots(figsize=(6,4))
                ax.bar(['ANN','SNN'], [ann_acc, snn_acc], color=['blue','red'], alpha=0.8)
                ax.set_ylabel('Final Accuracy')
                ax.set_title(f'{dataset.upper()} Final Accuracy')
                ax.set_ylim(0, max(0.2, max(ann_acc, snn_acc)*1.2, 0.1))
                for i,v in enumerate([ann_acc, snn_acc]):
                    ax.text(i, v+0.005, f'{v:.4f}', ha='center', fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(charts_path, f'appendix_{dataset}_final_accuracy.png'), dpi=300)
                plt.close()
                # 2) Inference Time
                fig, ax = plt.subplots(figsize=(6,4))
                ax.bar(['ANN','SNN'], [ann_time, snn_time], color=['blue','red'], alpha=0.8)
                ax.set_ylabel('Inference Time (s)')
                ax.set_title(f'{dataset.upper()} Inference Time')
                for i,v in enumerate([ann_time, snn_time]):
                    ax.text(i, v+1e-3, f'{v:.3f}', ha='center', fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(charts_path, f'appendix_{dataset}_inference_time.png'), dpi=300)
                plt.close()
                # 3) Energy per Sample (approx from total energy if per-sample not available)
                fig, ax = plt.subplots(figsize=(6,4))
                ax.bar(['ANN','SNN'], [ann_energy, snn_energy], color=['blue','red'], alpha=0.8)
                ax.set_ylabel('Energy (J)')
                ax.set_title(f'{dataset.upper()} Energy Consumption')
                for i,v in enumerate([ann_energy, snn_energy]):
                    ax.text(i, v+1e-2, f'{v:.2f}', ha='center', fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(charts_path, f'appendix_{dataset}_energy.png'), dpi=300)
                plt.close()
            except Exception as e:
                print(f"⚠️ Appendix chart generation failed for {dataset}: {e}")

def log_benchmark_results(results_file: str, save_dir: str = './results'):
    """
    PURPOSE: Log and save comprehensive benchmark results from file
    
    PARAMETERS:
    - results_file: Path to JSON file containing benchmark results
    - save_dir: Directory to save generated reports and tables
    
    PROCESSING STEPS:
    1. Load benchmark results from JSON file
    2. Create MetricsLogger instance
    3. Generate comprehensive summary report
    4. Create LaTeX table for research papers
    5. Save all outputs to specified directory
    
    OUTPUTS GENERATED:
    - Summary report with key findings and analysis
    - LaTeX table suitable for research paper inclusion
    - Comprehensive logging of all metrics
    - Structured data for further analysis
    
    USAGE:
    - Called after benchmark completion
    - Generates research-ready outputs
    - Provides comprehensive documentation
    - Enables further research analysis
    """
    # DEBUG: Check parameters
    print(f"🔧 DEBUG: log_benchmark_results called with results_file: {results_file}")
    print(f"🔧 DEBUG: save_dir: {save_dir}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # DEBUG: Check loaded results
    print(f"🔧 DEBUG: Loaded results type: {type(results)}")
    if isinstance(results, dict):
        print(f"🔧 DEBUG: Results keys: {list(results.keys())}")
    
    logger = MetricsLogger(save_dir)
    
    # Create summary report
    if 'results' in results:
        print(f"🔧 DEBUG: Found 'results' key, calling create_summary_report")
        try:
            logger.create_summary_report(results['results'])
            print(f"🔧 DEBUG: Calling create_latex_table")
            logger.create_latex_table(results['results'])
        except Exception as e:
            print(f"🔧 DEBUG: Error in create_summary_report/create_latex_table: {e}")
    else:
        print(f"🔧 DEBUG: No 'results' key found, keys available: {list(results.keys())}")
        # Handle the actual structure: {'snn': {...}, 'ann': {...}, 'comparison': {...}}
        if 'snn' in results and 'ann' in results:
            print(f"🔧 DEBUG: Found 'snn' and 'ann' keys, creating summary report")
            try:
                # Determine dataset key from results if available
                dataset_key = results.get('dataset') or results.get('current_dataset') or 'nmnist'
                dataset_key = str(dataset_key).lower()
                summary_data = {
                    dataset_key: {
                        'ann_accuracy': {'accuracy': results['ann'].get('final_accuracy', 0.0)},
                        'snn_accuracy': {'accuracy': results['snn'].get('final_accuracy', 0.0)},
                        'energy_comparison': {
                            'ann_inference': {
                                'total_energy_joules': results['ann'].get('metrics', {}).get('energy_consumption', {}),
                                'total_inference_time_s': results['ann'].get('inference_time', 0.0)
                            },
                            'snn_inference': {
                                'total_energy_joules': results['snn'].get('metrics', {}).get('energy_consumption', {}),
                                'total_inference_time_s': results['snn'].get('inference_time', 0.0)
                            }
                        },
                        'enhanced_metrics': results['snn'].get('enhanced_metrics', {})
                    }
                }
                # If config declares SHD but no results, add pending row
                try:
                    config_path = os.path.join(save_dir, 'config.json')
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as cf:
                            cfg = json.load(cf)
                        if 'shd' in [d.lower() for d in cfg.get('datasets', [])] and 'shd' not in summary_data:
                            summary_data['shd'] = {'pending': True}
                except Exception:
                    pass
                print(f"🔧 DEBUG: Summary data created: {summary_data}")
                logger.create_summary_report(summary_data)
                logger.create_latex_table(summary_data)
                # Create minimal appendix charts per dataset (if available)
                try:
                    logger.create_appendix_charts(summary_data)
                except Exception as e:
                    print(f"⚠️ Appendix charts skipped: {e}")
            except Exception as e:
                print(f"🔧 DEBUG: Error creating summary report: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"🔧 DEBUG: Unexpected structure, skipping summary report")
    
    print(f"All metrics logged to: {save_dir}") 
