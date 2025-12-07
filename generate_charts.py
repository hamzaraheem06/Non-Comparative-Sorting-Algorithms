"""
This script generates comprehensive performance visualization charts including:
- Theoretical vs Empirical complexity analysis
- Memory usage analysis
- Algorithm selection decision frameworks
- Distribution-specific performance analysis
- Advanced statistical analysis

Usage: python generate_charts.py [performance_data.csv]
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import os
from pathlib import Path


def setup_matplotlib():
    """Setup matplotlib with configuration."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['grid.alpha'] = 0.3

def enrich_csv_data(df):
    """Add theoretical complexity and other calculated columns to CSV data."""
    # Algorithm configurations with theoretical complexities
    algorithms = {
        'Counting Sort (Non-Stable)': {'theoretical': lambda n, k: n + k, 'space': lambda n, k: k},
        'Counting Sort (Stable)': {'theoretical': lambda n, k: n + k, 'space': lambda n, k: n + k},
        'Radix Sort (LSD)': {'theoretical': lambda n, k, d: d * (n + 10), 'space': lambda n, k: n + 10},
        'Bucket Sort': {'theoretical': lambda n, k: n + k, 'space': lambda n, k: n + k},
        'Flash Sort': {'theoretical': lambda n, k: n, 'space': lambda n, k: n},
        'Spread Sort': {'theoretical': lambda n, k: n + k if k/n < 10 else n * np.log2(n), 'space': lambda n, k: n}
    }
    
    theoretical_complexities = []
    range_size_ratios = []
    
    for _, row in df.iterrows():
        alg_name = row['Algorithm']
        n, k = row['ArraySize'], row['ValueRange']
        d = len(str(max(k, 1)))  # number of digits
        
        # Calculate theoretical complexity
        if alg_name in algorithms and 'theoretical' in algorithms[alg_name]:
            complexity_func = algorithms[alg_name]['theoretical']
            if 'd' in complexity_func.__code__.co_varnames:
                theo_complexity = complexity_func(n, k, d)
            else:
                theo_complexity = complexity_func(n, k)
        else:
            theo_complexity = n + k  # fallback
        
        theoretical_complexities.append(theo_complexity)
        range_size_ratios.append(k / n)
    
    # Add calculated columns
    df['TheoreticalComplexity'] = theoretical_complexities
    df['RangeSizeRatio'] = range_size_ratios
    
    return df

def create_advanced_sample_data():
    """Create comprehensive sample data including theoretical complexities and memory usage."""
    data = []
    
    # Algorithm configurations with theoretical complexities
    algorithms = {
        'Counting Sort (Non-Stable)': {'theoretical': lambda n, k: n + k, 'space': lambda n, k: k},
        'Counting Sort (Stable)': {'theoretical': lambda n, k: n + k, 'space': lambda n, k: n + k},
        'Radix Sort (LSD)': {'theoretical': lambda n, k, d: d * (n + 10), 'space': lambda n, k: n + 10},
        'Bucket Sort': {'theoretical': lambda n, k: n + k, 'space': lambda n, k: n + k},
        'Flash Sort': {'theoretical': lambda n, k: n, 'space': lambda n, k: n},
        'Spread Sort': {'theoretical': lambda n, k: n + k if k/n < 10 else n * np.log2(n), 'space': lambda n, k: n}
    }
    
    sizes = [100, 500, 1000, 2500, 5000]
    ranges = [50, 100, 500, 1000, 5000, 10000]
    distributions = ['uniform', 'normal', 'skewed', 'concentrated']
    
    for size in sizes:
        for range_val in ranges:
            for dist in distributions:
                # Calculate theoretical complexity
                n, k = size, range_val
                d = len(str(max(range_val, 1)))  # number of digits
                
                for alg_name, complexity in algorithms.items():
                    if 'theoretical' in complexity:
                        if 'd' in complexity['theoretical'].__code__.co_varnames:
                            theo_complexity = complexity['theoretical'](n, k, d)
                        else:
                            theo_complexity = complexity['theoretical'](n, k)
                    else:
                        theo_complexity = n + k  # fallback
                    
                    # Generate empirical performance with some noise
                    base_time = max(1, theo_complexity * 0.001)  # Scale factor
                    
                    # Add distribution-specific variations
                    dist_factor = 1.0
                    if dist == 'normal':
                        dist_factor = 1.2
                    elif dist == 'skewed':
                        dist_factor = 0.8
                    elif dist == 'concentrated':
                        dist_factor = 0.6
                    
                    empirical_time = base_time * dist_factor * (1 + np.random.normal(0, 0.1))
                    memory_used = complexity['space'](n, k) * 4  # 4 bytes per int
                    
                    data.append({
                        'Algorithm': alg_name,
                        'ArraySize': size,
                        'ValueRange': range_val,
                        'Distribution': dist,
                        'TheoreticalComplexity': theo_complexity,
                        'ExecutionTime_us': max(1, empirical_time),
                        'MemoryUsed_bytes': memory_used,
                        'RangeSizeRatio': k / n,
                        'EdgeCase': False
                    })
    
    return pd.DataFrame(data)

def create_theoretical_vs_empirical_chart(df, output_path):
    """Create theoretical vs empirical complexity comparison chart."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Theoretical vs Empirical Complexity Analysis', fontsize=16, fontweight='bold')
    
    algorithms = df['Algorithm'].unique()
    
    for idx, alg in enumerate(algorithms):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        alg_data = df[df['Algorithm'] == alg]
        
        # Group by array size and calculate means
        grouped = alg_data.groupby('ArraySize').agg({
            'TheoreticalComplexity': 'mean',
            'ExecutionTime_us': 'mean'
        }).reset_index()
        
        x = np.arange(len(grouped))
        width = 0.35
        
        ax.bar(x - width/2, grouped['TheoreticalComplexity'], width, 
               label='Theoretical Complexity', alpha=0.7, color='skyblue')
        ax.bar(x + width/2, grouped['ExecutionTime_us'], width, 
               label='Empirical Time (μs)', alpha=0.7, color='lightcoral')
        
        ax.set_title(alg, fontweight='bold')
        ax.set_xlabel('Array Size')
        ax.set_ylabel('Complexity / Time (μs)')
        ax.set_xticks(x)
        ax.set_xticklabels(grouped['ArraySize'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use log scale for better visualization
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_memory_usage_analysis(df, output_path):
    """Create memory usage analysis across algorithms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Memory Usage Analysis', fontsize=16, fontweight='bold')
    
    # Memory usage by algorithm
    memory_by_alg = df.groupby('Algorithm')['MemoryUsed_bytes'].mean()
    
    ax1.bar(range(len(memory_by_alg)), memory_by_alg.values, 
            color=sns.color_palette("husl", len(memory_by_alg)))
    ax1.set_title('Average Memory Usage by Algorithm')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Memory Usage (bytes)')
    ax1.set_xticks(range(len(memory_by_alg)))
    ax1.set_xticklabels(memory_by_alg.index, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Memory vs Array Size relationship
    for alg in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == alg]
        grouped = alg_data.groupby('ArraySize')['MemoryUsed_bytes'].mean()
        ax2.plot(grouped.index, grouped.values, marker='o', label=alg, linewidth=2)
    
    ax2.set_title('Memory Usage vs Array Size')
    ax2.set_xlabel('Array Size')
    ax2.set_ylabel('Memory Usage (bytes)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_algorithm_decision_framework(df, output_path):
    """Create algorithm selection decision framework visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create decision matrix based on range/size ratio and data characteristics
    ratios = df['RangeSizeRatio'].unique()
    distributions = df['Distribution'].unique()
    
    # Create heatmap data
    decision_matrix = np.zeros((len(distributions), len(ratios)))
    
    for i, dist in enumerate(distributions):
        for j, ratio in enumerate(ratios):
            subset = df[(df['Distribution'] == dist) & (df['RangeSizeRatio'] == ratio)]
            if not subset.empty:
                # Find best algorithm for this scenario
                best_alg = subset.loc[subset['ExecutionTime_us'].idxmin(), 'Algorithm']
                # Map algorithm to numeric score for visualization
                alg_scores = {
                    'Counting Sort (Non-Stable)': 1,
                    'Counting Sort (Stable)': 2,
                    'Radix Sort (LSD)': 3,
                    'Bucket Sort': 4,
                    'Flash Sort': 5,
                    'Spread Sort': 6
                }
                decision_matrix[i, j] = alg_scores.get(best_alg, 0)
    
    # Create heatmap
    sns.heatmap(decision_matrix, 
                xticklabels=[f'{r:.2f}' for r in sorted(ratios)],
                yticklabels=distributions,
                annot=True, 
                fmt='.0f',
                cmap='viridis',
                ax=ax,
                cbar_kws={'label': 'Recommended Algorithm (1-6)'})
    
    ax.set_title('Algorithm Selection Decision Framework', fontsize=14, fontweight='bold')
    ax.set_xlabel('Range/Size Ratio (k/n)')
    ax.set_ylabel('Data Distribution')
    
    # Add algorithm legend
    alg_labels = ['1: Counting (NS)', '2: Counting (S)', '3: Radix', '4: Bucket', '5: Flash', '6: Spread']
    ax.text(1.02, 1, '\n'.join(alg_labels), transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_scalability_analysis_advanced(df, output_path):
    """Create advanced scalability analysis with trend lines."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Scalability Analysis', fontsize=16, fontweight='bold')
    
    algorithms = df['Algorithm'].unique()
    colors = sns.color_palette("husl", len(algorithms))
    
    # Performance vs Array Size
    ax1 = axes[0, 0]
    for i, alg in enumerate(algorithms):
        alg_data = df[df['Algorithm'] == alg]
        grouped = alg_data.groupby('ArraySize')['ExecutionTime_us'].mean().reset_index()
        
        ax1.scatter(grouped['ArraySize'], grouped['ExecutionTime_us'], 
                   color=colors[i], label=alg, s=60, alpha=0.7)
        
        # Fit trend line
        z = np.polyfit(grouped['ArraySize'], grouped['ExecutionTime_us'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(grouped['ArraySize'].min(), grouped['ArraySize'].max(), 100)
        ax1.plot(x_trend, p(x_trend), color=colors[i], linestyle='--', alpha=0.8)
    
    ax1.set_title('Performance vs Array Size')
    ax1.set_xlabel('Array Size')
    ax1.set_ylabel('Execution Time (μs)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Performance vs Value Range
    ax2 = axes[0, 1]
    for i, alg in enumerate(algorithms):
        alg_data = df[df['Algorithm'] == alg]
        grouped = alg_data.groupby('ValueRange')['ExecutionTime_us'].mean().reset_index()
        
        ax2.plot(grouped['ValueRange'], grouped['ExecutionTime_us'], 
                marker='o', color=colors[i], label=alg, linewidth=2)
    
    ax2.set_title('Performance vs Value Range')
    ax2.set_xlabel('Value Range')
    ax2.set_ylabel('Execution Time (μs)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Distribution impact analysis
    ax3 = axes[1, 0]
    dist_performance = df.groupby(['Distribution', 'Algorithm'])['ExecutionTime_us'].mean().reset_index()
    dist_pivot = dist_performance.pivot(index='Distribution', columns='Algorithm', values='ExecutionTime_us')
    
    sns.heatmap(dist_pivot, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax3)
    ax3.set_title('Performance by Distribution')
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Distribution')
    
    # Efficiency ratio (theoretical vs empirical)
    ax4 = axes[1, 1]
    df['EfficiencyRatio'] = df['TheoreticalComplexity'] / df['ExecutionTime_us']
    
    for i, alg in enumerate(algorithms):
        alg_data = df[df['Algorithm'] == alg]
        grouped = alg_data.groupby('ArraySize')['EfficiencyRatio'].mean().reset_index()
        
        ax4.plot(grouped['ArraySize'], grouped['EfficiencyRatio'], 
                marker='s', color=colors[i], label=alg, linewidth=2)
    
    ax4.set_title('Algorithm Efficiency Ratio')
    ax4.set_xlabel('Array Size')
    ax4.set_ylabel('Theoretical/Empirical Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_by_size_chart(df, output_path):
    """Create performance comparison across different array sizes."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Group data by algorithm and array size
    grouped = df.groupby(['Algorithm', 'ArraySize'])['ExecutionTime_us'].mean().reset_index()
    
    # Create line plot for each algorithm
    algorithms = grouped['Algorithm'].unique()
    colors = sns.color_palette("husl", len(algorithms))
    
    for i, alg in enumerate(algorithms):
        alg_data = grouped[grouped['Algorithm'] == alg]
        ax.plot(alg_data['ArraySize'], alg_data['ExecutionTime_us'], 
                marker='o', linewidth=2.5, markersize=6, 
                color=colors[i], label=alg)
    
    ax.set_title('Performance Comparison Across Array Sizes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Array Size', fontsize=12)
    ax.set_ylabel('Execution Time (μs)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_heatmap(df, output_path):
    """Create heatmap visualization of algorithm performance."""
    # Create pivot table for heatmap
    pivot_table = df.pivot_table(
        values='ExecutionTime_us', 
        index='Algorithm', 
        columns='ArraySize', 
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'Execution Time (μs)'})
    plt.title('Performance Heatmap: Algorithm vs Array Size', fontsize=14, fontweight='bold')
    plt.xlabel('Array Size', fontsize=12)
    plt.ylabel('Algorithm', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_algorithm_comparison_chart(df, output_path):
    """Create direct algorithm comparison for specific scenarios."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Average performance comparison
    avg_performance = df.groupby('Algorithm')['ExecutionTime_us'].mean().sort_values()
    
    bars1 = ax1.bar(range(len(avg_performance)), avg_performance.values, 
                    color=sns.color_palette("viridis", len(avg_performance)))
    ax1.set_title('Average Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Average Execution Time (μs)')
    ax1.set_xticks(range(len(avg_performance)))
    ax1.set_xticklabels(avg_performance.index, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Performance variance analysis
    variance_data = df.groupby('Algorithm')['ExecutionTime_us'].agg(['mean', 'std']).reset_index()
    
    ax2.errorbar(range(len(variance_data)), variance_data['mean'], 
                yerr=variance_data['std'], fmt='o', capsize=5, capthick=2,
                color='darkblue', ecolor='lightblue', markersize=8)
    ax2.set_title('Performance Consistency Analysis', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Execution Time (μs) ± Std Dev')
    ax2.set_xticks(range(len(variance_data)))
    ax2.set_xticklabels(variance_data['Algorithm'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_scalability_analysis(df, output_path):
    """Create scalability analysis across different conditions."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Scalability Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance vs Array Size (log scale)
    grouped_size = df.groupby(['Algorithm', 'ArraySize'])['ExecutionTime_us'].mean().reset_index()
    
    for alg in grouped_size['Algorithm'].unique():
        alg_data = grouped_size[grouped_size['Algorithm'] == alg]
        ax1.loglog(alg_data['ArraySize'], alg_data['ExecutionTime_us'], 
                  marker='o', label=alg, linewidth=2)
    
    ax1.set_title('Scalability: Array Size Impact')
    ax1.set_xlabel('Array Size (log scale)')
    ax1.set_ylabel('Execution Time (μs, log scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance vs Value Range
    grouped_range = df.groupby(['Algorithm', 'ValueRange'])['ExecutionTime_us'].mean().reset_index()
    
    for alg in grouped_range['Algorithm'].unique():
        alg_data = grouped_range[grouped_range['Algorithm'] == alg]
        ax2.semilogx(alg_data['ValueRange'], alg_data['ExecutionTime_us'], 
                    marker='s', label=alg, linewidth=2)
    
    ax2.set_title('Scalability: Value Range Impact')
    ax2.set_xlabel('Value Range (log scale)')
    ax2.set_ylabel('Execution Time (μs)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribution impact
    dist_impact = df.groupby(['Distribution', 'Algorithm'])['ExecutionTime_us'].mean().reset_index()
    dist_pivot = dist_impact.pivot(index='Distribution', columns='Algorithm', values='ExecutionTime_us')
    
    sns.heatmap(dist_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3)
    ax3.set_title('Distribution Impact Analysis')
    ax3.set_xlabel('Algorithm')
    ax3.set_ylabel('Data Distribution')
    
    # 4. Efficiency over different sizes
    efficiency_data = []
    for alg in df['Algorithm'].unique():
        alg_data = df[df['Algorithm'] == alg]
        for size in alg_data['ArraySize'].unique():
            size_data = alg_data[alg_data['ArraySize'] == size]
            efficiency = size / size_data['ExecutionTime_us'].mean()  # elements per microsecond
            efficiency_data.append({'Algorithm': alg, 'ArraySize': size, 'Efficiency': efficiency})
    
    eff_df = pd.DataFrame(efficiency_data)
    eff_pivot = eff_df.pivot(index='ArraySize', columns='Algorithm', values='Efficiency')
    
    for col in eff_pivot.columns:
        ax4.plot(eff_pivot.index, eff_pivot[col], marker='d', label=col, linewidth=2)
    
    ax4.set_title('Algorithm Efficiency (Elements/μs)')
    ax4.set_xlabel('Array Size')
    ax4.set_ylabel('Elements Processed per μs')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    setup_matplotlib()
    
    print("Generating Performance Charts for Non-Comparison Sorting Algorithms")
    print("=" * 85)
    
    # Load or create data
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            print(f"Data loaded from {csv_file}")
            # Enrich with theoretical complexity data
            df = enrich_csv_data(df)
            print(f"Theoretical complexity data added")
        else:
            print(f"File {csv_file} not found, using sample data")
            df = create_advanced_sample_data()
    else:
        print("📖 Using comprehensive sample data for demonstration")
        df = create_advanced_sample_data()
    
    print(f"Data loaded: {len(df)} records")
    print(f"Array sizes: {sorted(df['ArraySize'].unique())}")
    print(f"Value ranges: {sorted(df['ValueRange'].unique())}")
    print(f"Distributions: {sorted(df['Distribution'].unique())}")
    print()
    
    # Create charts directory
    charts_dir = Path("charts")
    charts_dir.mkdir(exist_ok=True)
    
    # Generate all charts
    charts_generated = []
    
    # Original charts (enhanced)
    print("Generating enhanced performance_by_size.png...")
    create_performance_by_size_chart(df, charts_dir / "performance_by_size.png")
    charts_generated.append("performance_by_size.png")
    
    print("Generating enhanced performance_heatmap.png...")
    create_performance_heatmap(df, charts_dir / "performance_heatmap.png")
    charts_generated.append("performance_heatmap.png")
    
    print("Generating enhanced algorithm_comparison.png...")
    create_algorithm_comparison_chart(df, charts_dir / "algorithm_comparison.png")
    charts_generated.append("algorithm_comparison.png")
    
    print("Generating enhanced scalability_analysis.png...")
    create_scalability_analysis(df, charts_dir / "scalability_analysis.png")
    charts_generated.append("scalability_analysis.png")
    
    # New advanced charts
    print("Generating theoretical_vs_empirical.png...")
    create_theoretical_vs_empirical_chart(df, charts_dir / "theoretical_vs_empirical.png")
    charts_generated.append("theoretical_vs_empirical.png")
    
    print("Generating memory_usage_analysis.png...")
    create_memory_usage_analysis(df, charts_dir / "memory_usage_analysis.png")
    charts_generated.append("memory_usage_analysis.png")
    
    print("Generating algorithm_decision_framework.png...")
    create_algorithm_decision_framework(df, charts_dir / "algorithm_decision_framework.png")
    charts_generated.append("algorithm_decision_framework.png")
    
    print("Generating advanced_scalability_analysis.png...")
    create_scalability_analysis_advanced(df, charts_dir / "advanced_scalability_analysis.png")
    charts_generated.append("advanced_scalability_analysis.png")
    
    print()
    print("Advanced charts generated successfully in 'charts/' directory!")
    print("Generated charts:")
    for chart in charts_generated:
        print(f"   - {chart}")
    