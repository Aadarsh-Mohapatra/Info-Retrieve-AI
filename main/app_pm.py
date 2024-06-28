# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# General imports
import sys

sys.path.append("E:\\Github_Repo\\Info-Retrieve-AI")
from __init__ import cfg
import pandas as pd
import numpy as np
import requests
import time
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, sem, stats, t

# Streamlit and Pyngrok for web app
import streamlit as st
from pyngrok import ngrok

# Set up ngrok
ngrok.set_auth_token(cfg.NGROK_API_KEY)

# Read the Performance Metrics Datasets
data = {
    "GPT-4": pd.read_csv(
        r"E:\Github_Repo\Info-Retrieve-AI\ui_pm_output\ui_gpt4_performance_metrics.csv"
    ),
    "Gemini Pro": pd.read_csv(
        r"E:\Github_Repo\Info-Retrieve-AI\ui_pm_output\ui_gemini_performance_metrics.csv"
    ),
}


# Define functions for statistical analysis
def descriptive_stats(df, features):
    return df[features].describe()


def calculate_summary(df, features):
    return (
        df[features].mean().reset_index().rename(columns={"index": "Metric", 0: "Mean"})
    )


def calculate_bert_score(df):
    bert_score = (df["BERT_P"].mean() * df["BERT_R"].mean() * df["BERT_F1"].mean()) ** (
        1 / 3
    )
    return pd.DataFrame({"Metric": ["BERT Score"], "Mean": [bert_score]})


def compare_means(df1, df2, feature):
    data1 = df1[feature].dropna()
    data2 = df2[feature].dropna()
    stat, p_value = ttest_ind(data1, data2)
    df = len(data1) + len(data2) - 2
    pooled_std = np.sqrt(
        (
            (len(data1) - 1) * np.var(data1, ddof=1)
            + (len(data2) - 1) * np.var(data2, ddof=1)
        )
        / df
    )
    cohen_d = (np.mean(data1) - np.mean(data2)) / pooled_std
    return {"stat": stat, "p_value": p_value, "df": df, "cohen_d": cohen_d}


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data.dropna())
    n = len(a)
    m, se = np.mean(a), sem(a)
    h = se * t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h


def effect_size(d1, d2):
    return (d1.mean() - d2.mean()) / np.sqrt((d1.std() ** 2 + d2.std() ** 2) / 2)


# Define functions for Plots
def plot_kde_comparison(df1, df2, metrics, df1_name="GPT-4", df2_name="Gemini Pro"):
    sns.set_style("whitegrid")
    num_metrics = len(metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
    for i, metric in enumerate(metrics):
        sns.kdeplot(
            data=df1[metric].dropna(),
            ax=axs[i],
            fill=True,
            common_norm=False,
            palette="crest",
            label=df1_name,
        )
        sns.kdeplot(
            data=df2[metric].dropna(),
            ax=axs[i],
            fill=True,
            common_norm=False,
            palette="crest",
            label=df2_name,
        )
        axs[i].set_title(f"{metric} Distribution")
        axs[i].legend()
    plt.tight_layout()
    st.pyplot(fig)


def plot_box_comparison(df1, df2, metrics, df1_name="GPT-4", df2_name="Gemini Pro"):
    num_metrics = len(metrics)
    fig, axs = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 5))
    for i, metric in enumerate(metrics):
        combined_data = pd.concat(
            [df1[metric].rename(df1_name), df2[metric].rename(df2_name)], axis=1
        )
        sns.boxplot(data=combined_data, ax=axs[i])
        axs[i].set_title(f"{metric} Box Plot")
    plt.tight_layout()
    st.pyplot(fig)


def plot_heatmap(df, title):
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"shrink": 0.8}
    )
    plt.title(title)
    plt.tight_layout()
    st.pyplot(plt)


def plot_average_scores(df1, df2, metrics, labels):
    averages1 = [df1[metric].mean() for metric in metrics]
    averages2 = [df2[metric].mean() for metric in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, averages1, width, label=labels[0])
    rects2 = ax.bar(x + width / 2, averages2, width, label=labels[1])

    ax.set_ylabel("Scores")
    ax.set_title("Average scores by metric and model")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)


def plot_scatter_comparison(df, x_metric, y_metric, label):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_metric, y=y_metric)
    plt.title(f"{label}: {x_metric} vs {y_metric}")
    plt.tight_layout()
    st.pyplot(fig)


def plot_stacked_bar(df, metric, label):
    ranges = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    df["range"] = pd.cut(
        df[metric],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=ranges,
        include_lowest=True,
    )
    count_ranges = df["range"].value_counts(normalize=True).reindex(ranges)

    fig, ax = plt.subplots()
    count_ranges.plot(
        kind="bar", stacked=True, color=sns.color_palette("viridis", len(ranges))
    )
    plt.title(f"Distribution of {metric} scores in {label}")
    plt.ylabel("Proportion")
    plt.xlabel("Score Ranges")
    plt.tight_layout()
    st.pyplot(fig)


def plot_count_heatmap(df, metrics, label):
    count_data = pd.DataFrame()
    for metric in metrics:
        count_data[metric] = (
            pd.cut(df[metric], bins=5, include_lowest=True).value_counts().sort_index()
        )

    fig, ax = plt.subplots()
    sns.heatmap(count_data.T, annot=True, fmt=".0f", cmap="Blues")
    plt.title(f"Count of Scores by Range for {label}")
    plt.ylabel("Metrics")
    plt.xlabel("Score Ranges")
    plt.tight_layout()
    st.pyplot(fig)


# Streamlit UI for displaying performance metrics
st.title("Welcome to Info-Retrieve AI")
st.header("Performance Metrics")

# Model selection
models = st.sidebar.multiselect(
    "Select Models", ["GPT-4", "Gemini Pro"], default=["GPT-4", "Gemini Pro"]
)

# Metrics selection
metrics = [
    "BERT_F1",
    "BERT_R",
    "BERT_P",
    "BLEU",
    "ROGUE",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_relevancy",
    "context_recall",
    "context_entity_recall",
    "answer_similarity",
]
selected_metrics = st.sidebar.multiselect("Select Metrics", metrics, default=metrics)

# Display Statistics
if st.sidebar.checkbox("Show Statistics"):
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        [
            "Descriptive Statistics",
            "Summary",
            "Mean Confidence Interval",
            "T-Test",
            "Effect Sizes",
            "BERT Score",
        ],
    )

    for model in models:
        if analysis_type == "Descriptive Statistics":
            st.subheader(f"Descriptive Statistics for {model}:")
            st.write(data[model][selected_metrics].describe())

        elif analysis_type == "Summary":
            summary_data = data[model][selected_metrics].mean().reset_index(name="Mean")
            st.subheader(f"Summary of Selected Metrics for {model}")
            st.write(summary_data)

        elif analysis_type == "Mean Confidence Interval":
            st.subheader(f"Mean Confidence Intervals for {model}")
            for metric in selected_metrics:
                mean, lower, upper = mean_confidence_interval(data[model][metric])
                st.write(f"{metric} - Mean: {mean:.2f}, CI: [{lower:.2f}, {upper:.2f}]")

        elif analysis_type == "BERT Score":
            st.subheader(f"BERT Score for {model}:")
            bert_score = calculate_bert_score(data[model])
            st.write(bert_score)

    if len(models) == 2 and "T-Test" in analysis_type:
        st.subheader("T-Test Comparison Results")
        for metric in selected_metrics:
            results = compare_means(data["GPT-4"], data["Gemini Pro"], metric)
            st.write(
                f"T-test for '{metric}': Statistic = {results['stat']:.2f}, P-value = {results['p_value']:.4f}, Degrees of Freedom = {results['df']}, Cohen's d = {results['cohen_d']:.2f}"
            )

    if len(models) == 2 and "Effect Sizes" in analysis_type:
        st.subheader("Effect Sizes Comparison")
        for metric in selected_metrics:
            es = effect_size(data["GPT-4"][metric], data["Gemini Pro"][metric])
            st.write(f"Effect Size for {metric}: {es:.2f}")

# Display Plots
if st.sidebar.checkbox("Show Plots"):
    plot_type = st.sidebar.selectbox(
        "Choose Plot Type",
        [
            "KDE Plots",
            "Box Plots",
            "Correlation Heatmaps",
            "Average Scores",
            "Scatter Comparisons",
            "Stacked Bar Plots",
            "Count Heatmaps",
        ],
    )

    if plot_type == "KDE Plots":
        kde_metric = st.sidebar.multiselect(
            "Select Metrics for KDE Plot", selected_metrics, key="kde_metric"
        )
        if len(models) == 2:
            st.subheader(f"KDE Plots Comparison")
            plot_kde_comparison(
                data[models[0]][kde_metric],
                data[models[1]][kde_metric],
                kde_metric,
                df1_name=models[0],
                df2_name=models[1],
            )

    if plot_type == "Box Plots":
        box_metric = st.sidebar.multiselect(
            "Select Metrics for Box Plot", selected_metrics, key="box_metric"
        )
        if len(models) == 2:
            st.subheader(f"Box Plots Comparison")
            plot_box_comparison(
                data[models[0]][box_metric],
                data[models[1]][box_metric],
                box_metric,
                df1_name=models[0],
                df2_name=models[1],
            )

    elif plot_type == "Correlation Heatmaps":
        for model in models:
            st.subheader(f"{model} Correlation Heatmap")
            plot_heatmap(data[model][selected_metrics], f"{model} Correlation Heatmap")

    elif plot_type == "Average Scores":
        if len(models) == 2:
            plot_average_scores(
                data[models[0]][selected_metrics],
                data[models[1]][selected_metrics],
                selected_metrics,
                [models[0], models[1]],
            )

    elif plot_type == "Scatter Comparisons":
        x_metric = st.sidebar.selectbox(
            "Select X-axis Metric", selected_metrics, key="x_metric"
        )
        y_metric = st.sidebar.selectbox(
            "Select Y-axis Metric", selected_metrics, key="y_metric"
        )
        for model in models:
            st.subheader(f"{model} Scatter Comparison")
            plot_scatter_comparison(data[model], x_metric, y_metric, model)

    elif plot_type == "Stacked Bar Plots":
        bar_metric = st.sidebar.selectbox(
            "Select Metric for Stacked Bar Plot", selected_metrics, key="bar_metric"
        )
        for model in models:
            st.subheader(f"{model} Stacked Bar Plot")
            plot_stacked_bar(data[model], bar_metric, model)

    elif plot_type == "Count Heatmaps":
        for model in models:
            st.subheader(f"{model} Count Heatmap")
            plot_count_heatmap(data[model][selected_metrics], selected_metrics, model)
