# -*- coding: utf-8 -*-
"""
多模态数据集分析工具
用于分析 MOSI、MOSEI、SIMS 三个数据集的基本统计信息
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, dataset_name, dataset_path, output_path):
        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.analysis_results = {}
        
    def analyze_csv_labels(self, csv_file):
        """分析 CSV 标签文件"""
        print(f"\n{'='*60}")
        print(f"正在分析 {self.dataset_name} 数据集...")
        print(f"{'='*60}")
        
        # 读取 CSV 文件
        df = pd.read_csv(csv_file)
        
        results = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'column_info': {}
        }
        
        # 分析每列的信息
        for col in df.columns:
            col_info = {
                'dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique())
            }
            
            # 数值列的统计
            if df[col].dtype in ['int64', 'float64']:
                col_info['min'] = float(df[col].min()) if not pd.isna(df[col].min()) else None
                col_info['max'] = float(df[col].max()) if not pd.isna(df[col].max()) else None
                col_info['mean'] = float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                col_info['std'] = float(df[col].std()) if not pd.isna(df[col].std()) else None
            
            results['column_info'][col] = col_info
        
        # 分析划分情况
        if 'mode' in df.columns:
            mode_counts = df['mode'].value_counts().to_dict()
            results['split_distribution'] = {k: int(v) for k, v in mode_counts.items()}
            print(f"\n数据集划分:")
            for mode, count in mode_counts.items():
                print(f"  - {mode}: {count} 条")
        
        # 分析标签分布
        if 'label' in df.columns:
            label_stats = df['label'].describe()
            results['label_stats'] = {
                'count': int(label_stats['count']),
                'mean': float(label_stats['mean']),
                'std': float(label_stats['std']),
                'min': float(label_stats['min']),
                '25%': float(label_stats['25%']),
                '50%': float(label_stats['50%']),
                '75%': float(label_stats['75%']),
                'max': float(label_stats['max'])
            }
            print(f"\n标签统计:")
            print(f"  - 范围: [{label_stats['min']:.3f}, {label_stats['max']:.3f}]")
            print(f"  - 均值: {label_stats['mean']:.3f}")
            print(f"  - 标准差: {label_stats['std']:.3f}")
        
        # 多模态标签分析（SIMS数据集特有）
        multi_modal_cols = [col for col in df.columns if col.startswith('label_')]
        if multi_modal_cols:
            results['multi_modal_labels'] = {}
            print(f"\n多模态标签分析:")
            for col in multi_modal_cols:
                stats = df[col].describe()
                results['multi_modal_labels'][col] = {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max'])
                }
                print(f"  - {col}: 均值={stats['mean']:.3f}, 范围=[{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # 文本长度分析
        if 'text' in df.columns:
            text_lengths = df['text'].astype(str).apply(lambda x: len(x))
            results['text_length_stats'] = {
                'mean': float(text_lengths.mean()),
                'std': float(text_lengths.std()),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'median': float(text_lengths.median())
            }
            print(f"\n文本长度统计:")
            print(f"  - 平均长度: {text_lengths.mean():.1f} 字符")
            print(f"  - 长度范围: [{text_lengths.min()}, {text_lengths.max()}] 字符")
        
        # 情感类别分析（如果有 annotation 列）
        if 'annotation' in df.columns:
            annotation_counts = df['annotation'].value_counts().to_dict()
            results['annotation_distribution'] = {k: int(v) for k, v in annotation_counts.items()}
            print(f"\n情感类别分布:")
            for anno, count in annotation_counts.items():
                percentage = count / len(df) * 100
                print(f"  - {anno}: {count} 条 ({percentage:.1f}%)")
        
        return results, df
    
    def analyze_pickle_features(self, pickle_file):
        """分析 pickle 特征文件"""
        if not os.path.exists(pickle_file):
            print(f"  [警告] 特征文件不存在: {pickle_file}")
            return None
        
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            results = {}
            
            # 检查数据结构
            if isinstance(data, dict):
                results['keys'] = list(data.keys())
                print(f"\nPickle 文件结构:")
                print(f"  - 顶层键: {list(data.keys())}")
                
                # 分析每个划分的数据
                for split in ['train', 'valid', 'test', 'dev']:
                    if split in data or (split == 'dev' and 'valid' in data):
                        actual_split = split if split in data else 'valid'
                        split_data = data[actual_split]
                        
                        if isinstance(split_data, dict):
                            split_info = {
                                'sample_count': len(split_data.get('id', [])),
                                'feature_keys': list(split_data.keys())
                            }
                            
                            # 分析特征维度
                            if 'vision' in split_data and len(split_data['vision']) > 0:
                                vision_shape = split_data['vision'][0].shape if hasattr(split_data['vision'][0], 'shape') else split_data['vision'][0].__len__()
                                split_info['vision_shape'] = str(vision_shape)
                                split_info['vision_samples'] = len(split_data['vision'])
                            
                            if 'audio' in split_data and len(split_data['audio']) > 0:
                                audio_shape = split_data['audio'][0].shape if hasattr(split_data['audio'][0], 'shape') else split_data['audio'][0].__len__()
                                split_info['audio_shape'] = str(audio_shape)
                                split_info['audio_samples'] = len(split_data['audio'])
                            
                            if 'labels' in split_data:
                                split_info['labels_count'] = len(split_data['labels'])
                            
                            results[f'{split}_info'] = split_info
                            print(f"  - {split}: {split_info['sample_count']} 样本, 特征键: {split_info['feature_keys']}")
            
            return results
            
        except Exception as e:
            print(f"  [错误] 无法读取 pickle 文件: {e}")
            return None
    
    def visualize_label_distribution(self, df, save_path):
        """可视化标签分布"""
        if 'label' not in df.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        axes[0].hist(df['label'], bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('情感标签')
        axes[0].set_ylabel('频数')
        axes[0].set_title(f'{self.dataset_name} 标签分布直方图')
        axes[0].axvline(df['label'].mean(), color='r', linestyle='--', label=f'均值={df["label"].mean():.2f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 箱线图（按划分）
        if 'mode' in df.columns:
            df.boxplot(column='label', by='mode', ax=axes[1])
            axes[1].set_xlabel('数据集划分')
            axes[1].set_ylabel('情感标签')
            axes[1].set_title(f'{self.dataset_name} 各划分标签分布')
            plt.suptitle('')  # 移除自动生成的标题
        else:
            axes[1].boxplot(df['label'])
            axes[1].set_ylabel('情感标签')
            axes[1].set_title(f'{self.dataset_name} 标签箱线图')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  标签分布图已保存: {save_path}")
    
    def visualize_text_length(self, df, save_path):
        """可视化文本长度分布"""
        if 'text' not in df.columns:
            return
        
        text_lengths = df['text'].astype(str).apply(lambda x: len(x))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        axes[0].hist(text_lengths, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0].set_xlabel('文本长度（字符数）')
        axes[0].set_ylabel('频数')
        axes[0].set_title(f'{self.dataset_name} 文本长度分布')
        axes[0].axvline(text_lengths.mean(), color='r', linestyle='--', label=f'均值={text_lengths.mean():.1f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 按划分的文本长度
        if 'mode' in df.columns:
            for mode in df['mode'].unique():
                mode_lengths = df[df['mode'] == mode]['text'].astype(str).apply(lambda x: len(x))
                axes[1].hist(mode_lengths, bins=20, alpha=0.5, label=mode)
            axes[1].set_xlabel('文本长度（字符数）')
            axes[1].set_ylabel('频数')
            axes[1].set_title('各划分文本长度分布对比')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  文本长度分布图已保存: {save_path}")
    
    def visualize_multi_modal_comparison(self, df, save_path):
        """可视化多模态标签对比（仅 SIMS）"""
        multi_modal_cols = [col for col in df.columns if col.startswith('label_') and col != 'label']
        
        if len(multi_modal_cols) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # 各模态标签分布对比
        for i, col in enumerate(multi_modal_cols[:4]):  # 最多显示4个
            axes[i].hist(df[col], bins=20, alpha=0.7, label=col)
            axes[i].axvline(df[col].mean(), color='r', linestyle='--')
            axes[i].set_xlabel('标签值')
            axes[i].set_ylabel('频数')
            axes[i].set_title(f'{col} 分布 (均值={df[col].mean():.2f})')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  多模态标签对比图已保存: {save_path}")
    
    def save_report(self, results):
        """保存分析报告"""
        report_file = self.output_path / f'{self.dataset_name}_analysis_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存: {report_file}")
    
    def run_full_analysis(self):
        """运行完整分析"""
        print(f"\n{'#'*70}")
        print(f"# {self.dataset_name} 数据集完整分析")
        print(f"{'#'*70}")
        
        # 查找 CSV 文件
        csv_files = list(self.dataset_path.glob('*.csv'))
        if not csv_files:
            print(f"[错误] 未找到 CSV 文件在 {self.dataset_path}")
            return None
        
        csv_file = csv_files[0]  # 使用第一个 CSV 文件
        
        # 分析 CSV
        csv_results, df = self.analyze_csv_labels(csv_file)
        self.analysis_results['csv_analysis'] = csv_results
        
        # 查找 pickle 文件
        pickle_files = list(self.dataset_path.glob('*.pkl'))
        if pickle_files:
            pickle_file = [f for f in pickle_files if 'noalign' in f.name or 'data' in f.name]
            if pickle_file:
                pickle_results = self.analyze_pickle_features(pickle_file[0])
                if pickle_results:
                    self.analysis_results['pickle_analysis'] = pickle_results
        
        # 生成可视化
        print(f"\n正在生成可视化图表...")
        self.visualize_label_distribution(df, self.output_path / f'{self.dataset_name}_label_distribution.png')
        self.visualize_text_length(df, self.output_path / f'{self.dataset_name}_text_length.png')
        self.visualize_multi_modal_comparison(df, self.output_path / f'{self.dataset_name}_multi_modal_comparison.png')
        
        # 保存报告
        self.save_report(self.analysis_results)
        
        print(f"\n{'='*60}")
        print(f"{self.dataset_name} 分析完成!")
        print(f"{'='*60}")
        
        return self.analysis_results


def compare_datasets(results_dict, output_path):
    """对比多个数据集"""
    print(f"\n{'#'*70}")
    print(f"# 数据集对比分析")
    print(f"{'#'*70}")
    
    comparison = {
        'datasets': list(results_dict.keys()),
        'sample_counts': {},
        'label_ranges': {},
        'split_ratios': {}
    }
    
    for name, results in results_dict.items():
        if 'csv_analysis' in results:
            csv_info = results['csv_analysis']
            comparison['sample_counts'][name] = csv_info.get('total_samples', 0)
            
            if 'label_stats' in csv_info:
                comparison['label_ranges'][name] = {
                    'min': csv_info['label_stats']['min'],
                    'max': csv_info['label_stats']['max'],
                    'mean': csv_info['label_stats']['mean']
                }
            
            if 'split_distribution' in csv_info:
                total = sum(csv_info['split_distribution'].values())
                comparison['split_ratios'][name] = {
                    k: f"{v/total*100:.1f}%" 
                    for k, v in csv_info['split_distribution'].items()
                }
    
    # 打印对比结果
    print("\n【样本数量对比】")
    for name, count in comparison['sample_counts'].items():
        print(f"  {name}: {count} 条")
    
    print("\n【标签范围对比】")
    for name, range_info in comparison['label_ranges'].items():
        print(f"  {name}: [{range_info['min']:.2f}, {range_info['max']:.2f}], 均值={range_info['mean']:.2f}")
    
    print("\n【划分比例对比】")
    for name, ratios in comparison['split_ratios'].items():
        print(f"  {name}: {ratios}")
    
    # 保存对比报告
    comparison_file = Path(output_path) / 'datasets_comparison.json'
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"\n对比报告已保存: {comparison_file}")
    
    # 生成对比图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 样本数量对比
    names = list(comparison['sample_counts'].keys())
    counts = list(comparison['sample_counts'].values())
    axes[0].bar(names, counts, color=['skyblue', 'lightgreen', 'salmon'])
    axes[0].set_ylabel('样本数量')
    axes[0].set_title('各数据集样本数量对比')
    for i, v in enumerate(counts):
        axes[0].text(i, v, str(v), ha='center', va='bottom')
    
    # 标签均值对比
    if comparison['label_ranges']:
        means = [comparison['label_ranges'][n]['mean'] for n in names if n in comparison['label_ranges']]
        valid_names = [n for n in names if n in comparison['label_ranges']]
        axes[1].bar(valid_names, means, color=['skyblue', 'lightgreen', 'salmon'])
        axes[1].set_ylabel('标签均值')
        axes[1].set_title('各数据集情感均值对比')
        axes[1].axhline(0, color='r', linestyle='--', alpha=0.5)
    
    # 划分数量对比（堆叠柱状图）
    if comparison['split_ratios']:
        split_data = {}
        for name in names:
            if name in results_dict and 'csv_analysis' in results_dict[name]:
                split_dist = results_dict[name]['csv_analysis'].get('split_distribution', {})
                for split, count in split_dist.items():
                    if split not in split_data:
                        split_data[split] = []
                    split_data[split].append(count)
        
        bottom = np.zeros(len(names))
        colors = {'train': 'skyblue', 'valid': 'lightgreen', 'test': 'salmon', 'dev': 'orange'}
        for split, counts in split_data.items():
            if len(counts) == len(names):
                axes[2].bar(names, counts, bottom=bottom, label=split, color=colors.get(split, 'gray'))
                bottom += np.array(counts)
        axes[2].set_ylabel('样本数量')
        axes[2].set_title('各数据集划分对比')
        axes[2].legend()
    
    plt.tight_layout()
    comparison_chart = Path(output_path) / 'datasets_comparison.png'
    plt.savefig(comparison_chart, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"对比图表已保存: {comparison_chart}")


def main():
    """主函数"""
    # 设置路径
    base_dir = Path(__file__).parent.parent  # 项目根目录
    datasets_dir = base_dir / 'MCL-MCF' / 'datasets'
    output_dir = base_dir / 'outputs' / 'data_analysis'
    
    print(f"项目根目录: {base_dir}")
    print(f"数据集目录: {datasets_dir}")
    print(f"输出目录: {output_dir}")
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 要分析的数据集
    datasets = ['MOSI', 'MOSEI', 'SIMS']
    
    results_dict = {}
    
    for dataset_name in datasets:
        dataset_path = datasets_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"\n[跳过] 数据集目录不存在: {dataset_path}")
            continue
        
        # 创建分析器并运行分析
        analyzer = DatasetAnalyzer(dataset_name, dataset_path, output_dir)
        results = analyzer.run_full_analysis()
        
        if results:
            results_dict[dataset_name] = results
    
    # 如果有多个数据集，进行对比分析
    if len(results_dict) > 1:
        compare_datasets(results_dict, output_dir)
    
    print(f"\n{'='*70}")
    print("所有分析完成!")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
