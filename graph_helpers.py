import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

def plot_kde_with_threshold(data, threshold, output_file, title, xlabel, ylabel):
    """
    Plot a KDE graph with a vertical line marking a threshold.

    Args:
        data (pd.Series): Data to be plotted.
        threshold (float): Threshold value for the vertical line.
        output_file (str): Path to save the plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        fill_color (str): Color for the KDE plot fill.
        threshold_color (str): Color for the threshold line.
    """
    fill_color = "#A47864"
    # fill_color = "#5F9EA0"
    threshold_color="#D0052B"
    plt.figure(figsize=(16, 12))
    sns.kdeplot(data=data.dropna(), fill=True, color=fill_color, alpha=0.5)
    plt.axvline(x=threshold, color=threshold_color, linestyle='-')
    plt.text(
        x=threshold,
        y=plt.gca().get_ylim()[1] * 0.99,
        s=f'95th Percentile\n{threshold:.10f}',
        color=threshold_color,
        fontsize=12,
        ha='center',
        va='top',
        bbox=dict(boxstyle="round,pad=0.1", edgecolor='none', facecolor='white', alpha=0.7)
    )
    plt.title(title, fontsize=18, pad=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_bar_with_threshold(data, group_col, value_col, threshold, output_file, title, xlabel, ylabel):
    """
    Plot a bar graph with a horizontal line marking a threshold and vertical lines connecting bars to their labels.

    Args:
        data (pd.DataFrame): DataFrame containing group identifiers and correlation values.
        group_col (str): Column in `data` representing group identifiers.
        value_col (str): Column in `data` representing correlation values.
        threshold (float): Threshold value for the horizontal line.
        output_file (str): Path to save the plot.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        threshold_color (str): Color for the threshold line.
        colormap (str): Colormap name for the bars (default: "Blues").
    """
    data = data.sort_values(by=value_col, ascending=True)
    num_bars = len(data)
    # custom_colors = [
    #     '#8db3b3', '#8da0b3', '#8d8db3', '#8e96a1', '#8fbc8f',
    #     '#6a9fb5', '#6b8fb5', '#6c7fb5', '#6d8fa1', '#6eac8f',
    #     '#4a8fb5', '#4b7fb5', '#4c6fb5', '#4d7f91', '#4e9c8f',
    #     '#2a7fb5', '#2b6fb5', '#2c5fb5', '#2d6f81', '#2e8c8f',
    #     '#b0b0b0', '#a0a0a0', '#909090', '#808080', '#707070'
    # ]
    custom_colors = [
        "#9C8A7B", "#A97D64", "#B47155", "#C07D67", "#CAA38A",
        "#D4BAA5", "#E1D5C8", "#A68978", "#AF7565", "#B86856",
        "#C3766A", "#CA9988", "#D0B3A1", "#DAC8BA", "#997A6E",
        "#A26A5E", "#A95E4E", "#B26C61", "#BB8378", "#C2978E"
    ]
    custom_cmap = ListedColormap(custom_colors[:num_bars])
    shuffled_colors = np.random.permutation(custom_cmap.colors)
    threshold_color="#D0052B"
    fig, ax = plt.subplots(figsize=(30, 16))
    ax = sns.barplot(
        data=data,
        x=group_col,
        y=value_col,
        palette=shuffled_colors,
        hue=group_col,
        dodge=False,
        ax=ax
    )
    plt.axhline(y=threshold, color=threshold_color, linestyle='-', linewidth=2)
    plt.text(
        x=0.1,
        y=threshold,
        s=f'95th Percentile\n{threshold:.10f}',
        color=threshold_color,
        fontsize=16,
        ha='center',
        va='center',
        bbox=dict(boxstyle="round,pad=0.1", edgecolor='none', facecolor='white', alpha=0.7)
    )
    ymin = data[value_col].min() * 1.1
    plt.ylim(ymin, data[value_col].max() * 1.2)
    for i, bar in enumerate(ax.patches):
        bar_x = bar.get_x() + bar.get_width() / 2
        ax.plot(
            [bar_x, bar_x],
            [ymin, bar.get_height()],
            color='gray',
            linewidth=0.8,
            linestyle='--',
            zorder=0
        )
    plt.xticks(rotation=90, ha='center', fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=28, pad=28)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
