#!/usr/bin/env python3
"""
Fuzzy Membership Function Visualization

Generates publication-quality visualizations of fuzzy membership functions
for inclusion in the thesis.

Usage:
    python code/visualization/membership_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import os
import sys

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.fuzzy_system.production_engine import CalibratedOGDVariables

# Set style for publication
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color palette (accessible, colorblind-friendly)
COLORS = {
    'recent': '#2166ac',      # Blue
    'fairly_recent': '#67a9cf', 
    'moderate': '#d1e5f0',
    'fairly_old': '#fddbc7',
    'old': '#b2182b',         # Red
    
    'sparse': '#d73027',      # Red  
    'moderate_comp': '#fc8d59',
    'complete': '#91cf60',
    'very_complete': '#1a9850', # Green
    
    'very_few': '#7b3294',    # Purple
    'few': '#c2a5cf',
    'moderate_res': '#f7f7f7',
    'many': '#a6dba0',
    'rich': '#008837',        # Green
    
    'very_low': '#d73027',
    'low': '#fc8d59',
    'moderate_sim': '#fee090',
    'high': '#91bfdb',
    'very_high': '#4575b4',   # Blue
    
    'not_relevant': '#d73027',
    'low_rel': '#fc8d59',
    'moderate_rel': '#fee090',
    'high_rel': '#91bfdb',
    'very_relevant': '#4575b4'
}


def plot_recency_membership(save_path: str = None):
    """Plot recency membership functions with portal statistics overlay."""
    var = CalibratedOGDVariables.create_recency_variable()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.linspace(0, 4000, 1000)
    
    colors = ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#b2182b']
    
    for i, (term, mf) in enumerate(var.terms.items()):
        y = [mf.evaluate(xi) for xi in x]
        ax.plot(x, y, label=term.replace('_', ' ').title(), 
               linewidth=2.5, color=colors[i])
        ax.fill_between(x, y, alpha=0.15, color=colors[i])
    
    # Add calibration markers
    percentiles = [
        (223, 'P25', '#333'),
        (776, 'Median', '#333'),
        (1149, 'Mean', '#666'),
        (2731, 'P90', '#333')
    ]
    
    for val, label, color in percentiles:
        ax.axvline(x=val, color=color, linestyle='--', alpha=0.7, linewidth=1)
        ax.annotate(label, xy=(val, 0.95), xytext=(5, 0),
                   textcoords='offset points', fontsize=9,
                   rotation=90, va='top')
    
    ax.set_xlabel('Days Since Last Modification')
    ax.set_ylabel('Membership Degree (μ)')
    ax.set_title('Recency Linguistic Variable\n(Calibrated from opendata.swiss Statistics)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 1.05)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_completeness_membership(save_path: str = None):
    """Plot completeness membership functions."""
    var = CalibratedOGDVariables.create_completeness_variable()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.linspace(0, 1, 500)
    
    colors = ['#d73027', '#fc8d59', '#91cf60', '#1a9850']
    
    for i, (term, mf) in enumerate(var.terms.items()):
        y = [mf.evaluate(xi) for xi in x]
        ax.plot(x, y, label=term.replace('_', ' ').title(),
               linewidth=2.5, color=colors[i])
        ax.fill_between(x, y, alpha=0.15, color=colors[i])
    
    # Mark observed range
    ax.axvspan(0.50, 0.83, alpha=0.1, color='blue', 
              label='Observed Range (50-83%)')
    
    ax.set_xlabel('Completeness Ratio')
    ax.set_ylabel('Membership Degree (μ)')
    ax.set_title('Completeness Linguistic Variable\n(Based on DCAT-AP CH Field Analysis)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_resources_membership(save_path: str = None):
    """Plot resources membership functions."""
    var = CalibratedOGDVariables.create_resources_variable()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.linspace(0, 25, 500)
    
    colors = ['#7b3294', '#c2a5cf', '#f7f7f7', '#a6dba0', '#008837']
    
    for i, (term, mf) in enumerate(var.terms.items()):
        y = [mf.evaluate(xi) for xi in x]
        ax.plot(x, y, label=term.replace('_', ' ').title(),
               linewidth=2.5, color=colors[i])
        ax.fill_between(x, y, alpha=0.15, color=colors[i])
    
    # Add statistics markers
    markers = [(4, 'Median'), (6, 'Mean')]
    for val, label in markers:
        ax.axvline(x=val, color='#333', linestyle='--', alpha=0.7, linewidth=1)
        ax.annotate(label, xy=(val, 0.95), xytext=(5, 0),
                   textcoords='offset points', fontsize=9, rotation=90, va='top')
    
    ax.set_xlabel('Number of Resources')
    ax.set_ylabel('Membership Degree (μ)')
    ax.set_title('Resources Linguistic Variable\n(Calibrated from Resource Count Distribution)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_similarity_membership(save_path: str = None):
    """Plot similarity membership functions."""
    var = CalibratedOGDVariables.create_similarity_variable()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.linspace(0, 1, 500)
    
    colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']
    
    for i, (term, mf) in enumerate(var.terms.items()):
        y = [mf.evaluate(xi) for xi in x]
        ax.plot(x, y, label=term.replace('_', ' ').title(),
               linewidth=2.5, color=colors[i])
        ax.fill_between(x, y, alpha=0.15, color=colors[i])
    
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Membership Degree (μ)')
    ax.set_title('Similarity Linguistic Variable\n(Query-Document Match Score)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_relevance_output(save_path: str = None):
    """Plot relevance output membership functions."""
    var = CalibratedOGDVariables.create_relevance_variable()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.linspace(0, 1, 500)
    
    colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']
    
    for i, (term, mf) in enumerate(var.terms.items()):
        y = [mf.evaluate(xi) for xi in x]
        ax.plot(x, y, label=term.replace('_', ' ').title(),
               linewidth=2.5, color=colors[i])
        ax.fill_between(x, y, alpha=0.15, color=colors[i])
    
    ax.set_xlabel('Relevance Score')
    ax.set_ylabel('Membership Degree (μ)')
    ax.set_title('Relevance Output Variable\n(Aggregated Fuzzy Output)')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.tight_layout()
    return fig


def plot_all_variables_grid(save_path: str = None):
    """Create a 2x3 grid of all membership functions."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Variables to plot
    variables = [
        (CalibratedOGDVariables.create_recency_variable(), 
         np.linspace(0, 4000, 500), 'Days', 'Recency'),
        (CalibratedOGDVariables.create_completeness_variable(),
         np.linspace(0, 1, 500), 'Ratio', 'Completeness'),
        (CalibratedOGDVariables.create_resources_variable(),
         np.linspace(0, 25, 500), 'Count', 'Resources'),
        (CalibratedOGDVariables.create_similarity_variable(),
         np.linspace(0, 1, 500), 'Score', 'Similarity'),
        (CalibratedOGDVariables.create_relevance_variable(),
         np.linspace(0, 1, 500), 'Score', 'Relevance (Output)')
    ]
    
    # Color schemes
    color_schemes = [
        ['#2166ac', '#67a9cf', '#d1e5f0', '#fddbc7', '#b2182b'],
        ['#d73027', '#fc8d59', '#91cf60', '#1a9850', '#333'],
        ['#7b3294', '#c2a5cf', '#f7f7f7', '#a6dba0', '#008837'],
        ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4'],
        ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']
    ]
    
    for idx, (var, x_range, xlabel, title) in enumerate(variables):
        ax = axes.flat[idx]
        colors = color_schemes[idx]
        
        for i, (term, mf) in enumerate(var.terms.items()):
            y = [mf.evaluate(xi) for xi in x_range]
            color = colors[i % len(colors)]
            ax.plot(x_range, y, label=term.replace('_', ' ').title(),
                   linewidth=2, color=color)
            ax.fill_between(x_range, y, alpha=0.1, color=color)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('μ')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.set_ylim(0, 1.05)
    
    # Hide unused subplot
    axes.flat[5].axis('off')
    
    plt.suptitle('Fuzzy Linguistic Variables for OGD Ranking', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_inference_example(save_path: str = None):
    """Visualize a complete inference example."""
    from code.fuzzy_system.production_engine import create_ogd_fuzzy_engine
    
    engine = create_ogd_fuzzy_engine()
    
    # Example input
    inputs = {
        'recency': 100,      # Recent
        'completeness': 0.80,  # Complete
        'resources': 6,       # Moderate to many
        'similarity': 0.75    # High
    }
    
    result = engine.infer(inputs)
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid spec
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Input variables
    input_vars = [
        ('recency', CalibratedOGDVariables.create_recency_variable(), 
         inputs['recency'], np.linspace(0, 3000, 300)),
        ('completeness', CalibratedOGDVariables.create_completeness_variable(),
         inputs['completeness'], np.linspace(0, 1, 300)),
        ('resources', CalibratedOGDVariables.create_resources_variable(),
         inputs['resources'], np.linspace(0, 15, 300)),
        ('similarity', CalibratedOGDVariables.create_similarity_variable(),
         inputs['similarity'], np.linspace(0, 1, 300))
    ]
    
    for idx, (name, var, value, x_range) in enumerate(input_vars):
        ax = fig.add_subplot(gs[0, idx])
        
        for term, mf in var.terms.items():
            y = [mf.evaluate(xi) for xi in x_range]
            ax.plot(x_range, y, label=term, linewidth=1.5, alpha=0.7)
        
        # Mark input value
        ax.axvline(x=value, color='red', linestyle='--', linewidth=2)
        ax.plot(value, 0, 'rv', markersize=10)
        
        ax.set_title(f'{name.title()}: {value}')
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, 0.5, 1])
    
    # Fuzzification results
    ax_fuzz = fig.add_subplot(gs[1, :2])
    
    fuzz_data = []
    for name, var, value, _ in input_vars:
        memberships = var.fuzzify(value)
        for term, degree in memberships.items():
            if degree > 0:
                fuzz_data.append({'Variable': name, 'Term': term, 'μ': degree})
    
    if fuzz_data:
        import pandas as pd
        df = pd.DataFrame(fuzz_data)
        
        vars_list = df['Variable'].unique()
        x_pos = np.arange(len(df))
        
        bars = ax_fuzz.bar(x_pos, df['μ'], color='steelblue')
        ax_fuzz.set_xticks(x_pos)
        ax_fuzz.set_xticklabels([f"{r['Variable']}\n{r['Term']}" for _, r in df.iterrows()], 
                               fontsize=8, rotation=45, ha='right')
        ax_fuzz.set_ylabel('Membership Degree')
        ax_fuzz.set_title('Fuzzification Results')
        ax_fuzz.set_ylim(0, 1.1)
    
    # Active rules
    ax_rules = fig.add_subplot(gs[1, 2:])
    
    top_rules = result.get_top_rules(8)
    rule_ids = [f"R{r.rule.id}" for r in top_rules]
    strengths = [r.firing_strength for r in top_rules]
    
    bars = ax_rules.barh(rule_ids, strengths, color='darkorange')
    ax_rules.set_xlabel('Firing Strength')
    ax_rules.set_title('Top Active Rules')
    ax_rules.set_xlim(0, 1)
    
    # Output aggregation
    ax_output = fig.add_subplot(gs[2, 1:3])
    
    relevance_var = CalibratedOGDVariables.create_relevance_variable()
    x_out = np.linspace(0, 1, 300)
    
    for term, mf in relevance_var.terms.items():
        y = [mf.evaluate(xi) for xi in x_out]
        ax_output.plot(x_out, y, label=term, linewidth=1.5, alpha=0.5)
    
    # Mark defuzzified output
    ax_output.axvline(x=result.crisp_output, color='red', linestyle='--', linewidth=2,
                     label=f'Output: {result.crisp_output:.2f}')
    ax_output.fill_between([result.crisp_output-0.02, result.crisp_output+0.02], 
                          [0, 0], [1, 1], alpha=0.3, color='red')
    
    ax_output.set_xlabel('Relevance Score')
    ax_output.set_ylabel('Membership')
    ax_output.set_title(f'Defuzzified Output: {result.crisp_output:.3f} ({result.crisp_output*100:.1f}%)')
    ax_output.legend(loc='upper right', fontsize=8)
    ax_output.set_ylim(0, 1.1)
    
    plt.suptitle('Mamdani Fuzzy Inference Example', fontsize=14, y=0.98)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def main():
    """Generate all publication figures."""
    output_dir = "figures/fuzzy_system"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating membership function plots...")
    
    plot_recency_membership(f"{output_dir}/mf_recency.png")
    plot_completeness_membership(f"{output_dir}/mf_completeness.png")
    plot_resources_membership(f"{output_dir}/mf_resources.png")
    plot_similarity_membership(f"{output_dir}/mf_similarity.png")
    plot_relevance_output(f"{output_dir}/mf_relevance.png")
    plot_all_variables_grid(f"{output_dir}/mf_all_variables.png")
    
    print("\nGenerating inference example...")
    plot_inference_example(f"{output_dir}/inference_example.png")
    
    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
