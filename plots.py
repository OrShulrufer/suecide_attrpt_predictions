from external_libs_imports import *

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Enhanced aesthetics
import scipy.stats as stats
import seaborn as sns


# Function to calculate the row-wise mean of columns with the same prefix
def calculate_rowwise_means(df, feature_prefixes):
    mean_series_list = []
    for prefix in feature_prefixes:
        relevant_cols = [col for col in df.columns if col[:re.search("\d+", col).start() - 1] == prefix]
        mean_series = df[relevant_cols].mean(axis=1)
        mean_series.name = f"{prefix}_mean"
        mean_series_list.append(mean_series)
    return pd.concat(mean_series_list, axis=1)



# Function to generate box plots along with statistics
def generate_box_plots_with_statistics(df_box, feature_prefix, palette='husl', no_zeros=True):
    folder_path = 'Data Plots/Box Plots'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    sns.set(style="whitegrid")

    df_box = df_box[df_box!=0]

    palette = sns.color_palette(palette, len(df_box.columns))
    fig, (ax_stats, ax_boxplots) = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df_box, ax=ax_boxplots, palette=palette)

    statistics = []
    for i, column in enumerate(df_box.columns):
        q1, q3 = df_box[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = len(df_box[(df_box[column] < q1 - 1.5 * iqr) | (df_box[column] > q3 + 1.5 * iqr)])
        stat_text = f'{column}:\nQ1: {q1:.2f}\nQ3: {q3:.2f}\nIQR: {iqr:.2f}\nOutliers: {outliers}'
        statistics.append(stat_text)
        ax_stats.text(0.1, 0.9 - i * 0.22, stat_text, fontsize=10, verticalalignment='center', color=palette[i])

    ax_stats.axis('off')
    ax_boxplots.set_title('Box Plots: without zeros')
    ax_boxplots.set_xticklabels(ax_boxplots.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(f"{folder_path}/{feature_prefix}_BOXPlot.png", bbox_inches='tight', replace=True)
    plt.close()


def plot_histogram_like_bar(df, title="Histogram-like Bar Plot of Multiple Features"):
    fig, ax = plt.subplots(figsize=(16, 8))  # Adjust the figure size

    # Define bin edges
    bin_edges = list(range(0, int(df.max().max()) + 2))

    # Calculate the number of columns
    num_columns = len(df.columns)

    # Calculate dynamic bar widths
    fixed_width = 1.0  # You can adjust this value
    bar_widths = [fixed_width / num_columns] * num_columns

    # Generate colors from the hls palette
    colors = sns.color_palette("hls", num_columns)

    # Loop through each feature to plot
    for i, column in enumerate(df.columns):
        # Calculate histogram-like values
        hist_values = df[column].value_counts().sort_index()

        # Fill in missing indices with zero frequency
        hist_values = hist_values.reindex(bin_edges, fill_value=0)

        # Calculate bar positions
        bar_positions = hist_values.index + (i * bar_widths[i])

        # Plot bars
        ax.bar(bar_positions, hist_values, width=bar_widths[i], alpha=0.5, label=f"{column}", edgecolor="black",
               linewidth=1.2, color=colors[i])

    # Set custom labels and titles
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    # Add legend at the bottom center
    ax.legend(loc='lower center')

    # Set custom labels at the bottom
    ax.set_xticks(bin_edges)
    ax.set_xticklabels([f"{i}" for i in bin_edges])

    plt.tight_layout()

    return fig, ax


def generate_qq_plot_for_means(df, feature_prefix):
    # Filter columns related to the feature
    feature_cols = [col for col in df.columns if col.startswith(feature_prefix)]

    # Calculate the mean across periods for each feature
    means = df[feature_cols].mean(axis=1)

    sns.set(style="whitegrid")
    sns.set_palette("husl")

    # Create QQ plot
    fig, ax = plt.subplots()
    stats.probplot(means, dist="norm", plot=ax)
    ax.set_title(f"QQ Plot for mean of {feature_prefix} across periods")

    # Close the plot to prevent it from displaying
    plt.close(fig)

    return fig


# Main function to generate different types of plots
def generate_plots(df_plot,
                   feature_prefixes,
                   plot_types=['boxplot', 'histogram', 'scatterplot', 'qq_plot', 'heatmap', 'pairplot'],
                   hue_column_name_for_pairplot=None) :

    for prefix in feature_prefixes:
        relevant_cols = [col for col in df_plot.columns if  col[:re.search("\d+", col).start() - 1] == prefix]

        if 'boxplot' in plot_types:
            generate_box_plots_with_statistics(df_plot[relevant_cols], prefix)

        if 'histogram' in plot_types:
            folder_path = 'Data Plots/Histograms'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            fig, ax = plot_histogram_like_bar(df_plot[relevant_cols], title="Histogram-like Bar Plot of One Featcher Over Time Piriods")
            fig.savefig(f"{folder_path}/{prefix}_Histogram.png", replace=True)
            plt.close()

        if 'scatterplot' in plot_types:
            folder_path = 'Data Plots/Scatter Plots'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            sns.pairplot(df_plot[relevant_cols], kind='scatter')
            plt.savefig(f"{folder_path}/{prefix}_ScatterPlot.png", replace=True)
            plt.close()

        if 'qq_plot' in plot_types:
            folder_path = 'Data Plots/QQ Plots'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            fig = generate_qq_plot_for_means(df_plot[relevant_cols], prefix)
            fig.savefig(f"{folder_path}/{prefix}_QQPlot.png", replace=True)
            plt.close()

        if 'heatmap' in plot_types:
            folder_path = 'Data Plots/Heatmaps'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            plt.figure(figsize=(16, 12))
            sns.heatmap(df_plot[relevant_cols].corr(), annot=True, cmap='coolwarm')
            plt.savefig(f"{folder_path}/{prefix}_Heatmap.png", replace=True)
            plt.close()

    if len(feature_prefixes) > 1:
        df_rowwise_means = calculate_rowwise_means(df_plot, feature_prefixes)

        if 'pairplot' in plot_types:
            folder_path = 'Data Plots/Hue_Pair Plots'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            if hue_column_name_for_pairplot==None:
                for col in df_rowwise_means.columns.tolist():
                    sns.pairplot(df_rowwise_means, palette='coolwarm', hue=col)
                    plt.savefig(f"{folder_path}/{col}_hue_Pairplot.png", replace=True)
                    plt.close()

        if 'histogram' in plot_types:
            folder_path = 'Data Plots/Histograms'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            fig, ax = plot_histogram_like_bar(df_rowwise_means, title="Histogram-like Bar Plot of One Featcher With Mean Of  Piriods")
            fig.savefig(f"{folder_path}/Time_Mean_Histogram.png", replace=True)
            plt.close()


def save_plot(classifier_name, plot_type, plt):
    folder = f'./Resolt_Plots/{classifier_name}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.savefig(f"{folder}/{plot_type}.png")
    plt.close()

def plot_feature_importance(classifier_name, feature_importance):
    sns.set_style("whitegrid")
    # Ensure importance_series is a Pandas Series
    if not isinstance(feature_importance, pd.Series):
        feature_importance = pd.Series(feature_importance)
    # Sort the series
    sorted_series = feature_importance.sort_values(ascending=False)

    sns.barplot(x=sorted_series.index, y=sorted_series, palette="husl")
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title(f"{classifier_name} - Feature Importance")
    save_plot(classifier_name, 'feature_importance', plt)


def plot_pie(classifier_name, importance_series):
    # Ensure importance_series is a Pandas Series
    if not isinstance(importance_series, pd.Series):
        importance_series = pd.Series(importance_series)
    # Sort the series
    sorted_series = importance_series.sort_values(ascending=False)
    # Select top 10 features
    top_10 = sorted_series.iloc[:10]
    # Sum up the other features' importances
    others = pd.Series([sorted_series.iloc[10:].sum()], index=['Others'])
    # Combine top 10 with others
    final_series = top_10.append(others)
    # Plot
    plt.figure(figsize=(7, 7))
    plt.pie(final_series, labels=final_series.index, autopct='%1.1f%%', startangle=90)
    plt.title(f"{classifier_name} - Top 10 Feature Importance (Pie Chart)")
    save_plot(classifier_name, 'pie_chart', plt)

def plotting_results(classeifire_metric_dict):
    for classifier_name, metrics in classeifire_metric_dict.items():
        if 'feature_importance' in metrics:
            plot_feature_importance(classifier_name, metrics['feature_importance'])
            plot_pie(classifier_name,  metrics['feature_importance'])
        plot_calibration(classifier_name, metrics)

    plot_roc_curve(classeifire_metric_dict)
    plot_all_metrics_bar_chart(classeifire_metric_dict)
    plot_permutation(classeifire_metric_dict)


def plot_permutation(classifier_metric_dict):
    sns.set_style("whitegrid")
    # Initialize empty DataFrame to hold graph data
    graph_data = pd.DataFrame()
    for clf_name, metrics in classifier_metric_dict.items():
        # Extract feature importance
        perm_importance = metrics['perm_importance']
        # Check if 'importances_mean' exists
        if 'importances_mean' in perm_importance:
            perm_mean = perm_importance['importances_mean']
        else:
            continue  # Skip to next iteration if 'importances_mean' is not present
        # Create a DataFrame row
        row_data = pd.DataFrame([perm_mean], columns=metrics['X_test'].columns)
        row_data['model_name'] = clf_name
        # Append to main DataFrame
        graph_data = pd.concat([graph_data, row_data], ignore_index=True)
    graph_data = graph_data.melt(id_vars='model_name', var_name='variable', value_name='value')

    # Create the plot
    plt.figure(figsize=[14, 5])
    plt.axhline(0, c='black')
    sns.barplot(x='variable', y='value', hue='model_name', data=graph_data)
    plt.title("Permutation Feature Importance Across Models")
    plt.xlabel("Variable Name")
    plt.ylabel("Metric Change")
    save_plot('all', 'permutation', plt)


def plot_calibration(classifier_name, metrics):
    sns.set_style("whitegrid")
    # Plot the Probabilities Calibrated curve
    plt.plot(metrics['fpr'],
             metrics['tpr'],
             marker='o',
             linewidth=1,
             label='Logistic Regression')
    # Plot the Perfectly Calibrated by Adding the 45-degree line to the plot
    plt.plot([0, 1],
             [0, 1],
             linestyle='--',
             label='Perfectly Calibrated')
    # Set the title and axis labels for the plot
    plt.title('Probability Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    # Add a legend to the plot
    plt.legend(loc='best')
    save_plot(classifier_name, 'calibration', plt)



def plot_all_metrics_bar_chart(metrics_dict):
    sns.set_style("whitegrid")

    df_dict = pd.DataFrame(metrics_dict)
    df_metrics = df_dict.T[['f1_score', 'f1_opt_t', 'recall','recall_opt_t', 'precision','precision_opt_t',
                            'balanced_accuracy','balanced_accuracy_opt_t', 'accuracy','accuracy_opt_t', 'brier']]

    sns.set_palette("husl", len(['f1_score', 'f1_opt_t', 'recall','recall_opt_t', 'precision','precision_opt_t',
                            'balanced_accuracy','balanced_accuracy_opt_t', 'accuracy','accuracy_opt_t', 'brier']))
    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 7))
    # Number of classifiers and metrics
    n_classifiers = len(df_metrics.index)
    n_metrics = len(df_metrics.columns)
    # Width of a bar group
    width = 0.03
    # Set the positions and width for the bars
    positions = np.arange(n_classifiers)
    # Plotting each metric's bar
    for i, metric in enumerate(df_metrics.columns):
        ax.bar(
            positions + i * width,  # position adjusted for each metric
            df_metrics[metric],  # data
            width,  # width of the bars
            label=metric  # label for the legend
        )
    # Setting the X axis labels and title
    ax.set_xlabel('Classifiers')
    ax.set_title('Metrics Comparison Across Classifiers by classifier')
    ax.set_xticks(positions + width * (n_metrics / 2) - width / 2)  # position x-ticks in the center of the grouped bars
    ax.set_xticklabels(df_metrics.index, rotation=45, ha='right')
    ax.legend()  # display the legend
    plt.grid(True)
    plt.tight_layout()
    save_plot('all', 'bars', plt)


def plot_roc_curve(metrics_dict):
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", len(metrics_dict))
    for i, (clf_name, metrics) in enumerate(metrics_dict.items()):
        plt.plot(metrics['fpr'], metrics['tpr'], color=palette[i],
                 label=f'{clf_name} (area = {metrics["auc_score"]:.2f})')


    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)
    save_plot('all', 'roc_curve', plt)






