import json
import os
import numpy as np
from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional

class ModelStats:
    def __init__(self, stats_dir: str = "../stats"):
        self.stats_dir = stats_dir
        self.stats_file = os.path.join(stats_dir, "model_stats.json")
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

    def load_sentiment_metrics(self, model_dir: str) -> Dict:
        metrics = {
            'accuracy': 0,
            'macro_f1': 0,
            'class_metrics': {
                'ignored': 0,
                'viewed': 0,
                'liked': 0
            }
        }

        test_results_path = os.path.join(model_dir, "test_results.json")

        if os.path.exists(test_results_path):
            with open(test_results_path, 'r') as f:
                loaded_metrics = json.load(f)

                metrics['accuracy'] = loaded_metrics.get('eval_accuracy', 0)
                metrics['macro_f1'] = loaded_metrics.get('eval_f1', 0)

                metrics['class_metrics'] = {
                    'ignored': loaded_metrics.get('eval_ignored_f1', 0),
                    'viewed': loaded_metrics.get('eval_viewed_f1', 0),
                    'liked': loaded_metrics.get('eval_liked_f1', 0)
                }

                total_f1 = sum(metrics['class_metrics'].values())

                if total_f1 > 0:
                    metrics['ignored_f1_ratio'] = metrics['class_metrics']['ignored'] / total_f1
                    metrics['viewed_f1_ratio'] = metrics['class_metrics']['viewed'] / total_f1
                    metrics['liked_f1_ratio'] = metrics['class_metrics']['liked'] / total_f1
                else:
                    metrics['ignored_f1_ratio'] = 0
                    metrics['viewed_f1_ratio'] = 0
                    metrics['liked_f1_ratio'] = 0

                metrics['class_balance_score'] = 1 - (
                        abs(metrics['ignored_f1_ratio'] - 0.33) +
                        abs(metrics['viewed_f1_ratio'] - 0.33) +
                        abs(metrics['liked_f1_ratio'] - 0.33)
                ) / 2

        return metrics

    def load_recommender_metrics(self, model_dir: str) -> Dict:
        metrics = {
            'rmse': 0,
            'ndcg@10': 0,
            'precision@10': 0,
            'recall@10': 0,
            'f1@10': 0,
            'coverage': 0
        }

        model_files = [f for f in os.listdir(model_dir) if f.startswith('metrics_') and f.endswith('.json')]

        if model_files:
            latest_metrics = max(model_files)
            metrics_path = os.path.join(model_dir, latest_metrics)

            with open(metrics_path, 'r') as f:
                loaded_metrics = json.load(f)

                metrics.update({
                    'rmse': loaded_metrics.get('rmse', 0),
                    'ndcg@10': loaded_metrics.get('ndcg@10', 0),
                    'precision@10': loaded_metrics.get('precision@10', 0),
                    'recall@10': loaded_metrics.get('recall@10', 0)
                })

                if metrics['precision@10'] > 0 and metrics['recall@10'] > 0:
                    metrics['f1@10'] = (
                            2 * metrics['precision@10'] * metrics['recall@10'] /
                            (metrics['precision@10'] + metrics['recall@10'])
                    )

            if os.path.exists(os.path.join(model_dir, 'model_components.json')):
                try:
                    with open(os.path.join(model_dir, 'model_components.json'), 'r') as f:
                        components = json.load(f)
                        total_items = len(components.get('product_mapping', {}))
                        metrics['coverage'] = min(10 / total_items, 1.0) if total_items > 0 else 0
                except (IOError, json.JSONDecodeError):
                    pass

        return metrics

    def load_score_metrics(self, model_dir: str) -> Dict:
        metrics = {}
        score_files = [f for f in os.listdir(model_dir) if f.startswith('score_metrics_') and f.endswith('.json')]

        if score_files:
            latest_metrics = max(score_files)
            metrics_path = os.path.join(model_dir, latest_metrics)

            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

        return metrics

    def update_stats(self) -> None:
        current_time = datetime.now().isoformat()

        stats = {
            'timestamp': current_time,
            'sentiment_analysis': {
                'accuracy': 0,
                'macro_f1': 0,
                'class_metrics': {
                    'ignored': 0,
                    'viewed': 0,
                    'liked': 0
                },
                'class_balance_score': 0
            },
            'recommender': {
                'rmse': 0,
                'ndcg@10': 0,
                'precision@10': 0,
                'recall@10': 0,
                'f1@10': 0,
                'coverage': 0
            }
        }

        sentiment_metrics = self.load_sentiment_metrics("../model/sns_sentiment_model")
        recommender_metrics = self.load_recommender_metrics("../model/recommender_model")
        score_metrics = self.load_score_metrics("../model/recommender_model")

        stats['sentiment_analysis'].update(sentiment_metrics)
        stats['recommender'].update(recommender_metrics)

        if score_metrics:
            stats['score_formula'] = {
                'distribution': {
                    'entropy': score_metrics.get('score_entropy', 0),
                    'skewness': score_metrics.get('score_skewness', 0),
                    'kurtosis': score_metrics.get('score_kurtosis', 0)
                },
                'correlations': {
                    'reaction_sentiment': score_metrics.get('reaction_sentiment_corr', 0),
                    'reaction_consistency': score_metrics.get('reaction_consistency_corr', 0),
                    'sentiment_consistency': score_metrics.get('sentiment_consistency_corr', 0)
                },
                'contributions': {
                    'reaction': score_metrics.get('reaction_contribution', 0),
                    'sentiment': score_metrics.get('sentiment_contribution', 0),
                    'consistency': score_metrics.get('consistency_contribution', 0)
                },
                'consistency': {
                    'rate': score_metrics.get('consistency_rate', 0),
                    'good_liked': score_metrics.get('good_liked_rate', 0),
                    'bad_ignored': score_metrics.get('bad_ignored_rate', 0)
                }
            }

        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                existing_stats = json.load(f)
            if isinstance(existing_stats, list):
                existing_stats.append(stats)
                stats = existing_stats
            else:
                stats = [existing_stats, stats]
        else:
            stats = [stats]

        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    st.set_page_config(page_title="Model Performance Dashboard", layout="wide")
    st.title("Model Performance Dashboard")

    stats = ModelStats()

    if st.button("Refresh Metrics"):
        stats.update_stats()

    if not os.path.exists(stats.stats_file):
        st.warning("No statistics data available. Click 'Refresh Metrics' button.")
        return

    with open(stats.stats_file, 'r') as f:
        all_stats = json.load(f)

    latest_stats = all_stats[-1]

    sentiment_stats = latest_stats.get('sentiment_analysis', {})
    recommender_stats = latest_stats.get('recommender', {})

    col1, col2 = st.columns(2)

    with col1:
        st.header("Sentiment Analysis Model")

        fig_accuracy = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_stats.get('accuracy', 0),
            title={'text': "Accuracy"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1]}}
        ))
        st.plotly_chart(fig_accuracy)

        class_metrics = sentiment_stats.get('class_metrics', {})
        fig_f1 = go.Figure(data=[
            go.Bar(
                x=list(class_metrics.keys()),
                y=list(class_metrics.values()),
                text=[f'{val:.3f}' for val in class_metrics.values()],
                textposition='auto',
            )
        ])
        fig_f1.update_layout(title="Class F1 Scores")
        st.plotly_chart(fig_f1)

        st.metric(
            "Class Balance Score",
            f"{sentiment_stats.get('class_balance_score', 0):.3f}"
        )

    with col2:
        st.header("Recommendation System")

        rec_metrics = latest_stats.get('recommender', {})

        cols = st.columns(2)
        cols[0].metric("RMSE", f"{rec_metrics.get('rmse', 0):.3f}")
        cols[1].metric("NDCG@10", f"{rec_metrics.get('ndcg@10', 0):.3f}")

        fig_pr = go.Figure(data=go.Scatter(
            x=[rec_metrics.get('recall@10', 0)],
            y=[rec_metrics.get('precision@10', 0)],
            mode='markers+text',
            text=['Current'],
            textposition='top center'
        ))
        fig_pr.update_layout(
            title="Precision-Recall Trade-off",
            xaxis_title="Recall@10",
            yaxis_title="Precision@10",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig_pr)

        fig_coverage = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rec_metrics.get('coverage', 0),
            title={'text': "Item Coverage"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1]}}
        ))
        st.plotly_chart(fig_coverage)

    st.header("Performance Trends")
    if len(all_stats) > 1:
        df = pd.DataFrame(all_stats)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig_sentiment_trend = go.Figure()
        fig_sentiment_trend.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['sentiment_analysis'].apply(lambda x: x.get('accuracy', 0)),
            name='Accuracy'
        ))
        fig_sentiment_trend.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['sentiment_analysis'].apply(lambda x: x.get('macro_f1', 0)),
            name='Macro F1'
        ))
        fig_sentiment_trend.update_layout(title="Sentiment Analysis Performance Trend")
        st.plotly_chart(fig_sentiment_trend)

        fig_recommender_trend = go.Figure()
        fig_recommender_trend.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['recommender'].apply(lambda x: x.get('ndcg@10', 0)),
            name='NDCG@10'
        ))
        fig_recommender_trend.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['recommender'].apply(lambda x: x.get('f1@10', 0)),
            name='F1@10'
        ))
        fig_recommender_trend.update_layout(title="Recommender System Performance Trend")
        st.plotly_chart(fig_recommender_trend)
    else:
        st.info("Not enough data for trend analysis.")

    st.header("Score Formula Analysis")

    if 'score_formula' not in latest_stats:
        st.warning("Score formula metrics not yet collected. Click 'Refresh Metrics' button.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Score Distribution")

        score_dist = latest_stats.get('score_formula', {}).get('distribution', {})
        if not score_dist:
            st.info("No distribution data available.")
        else:
            fig_entropy = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_dist['entropy'],
                title={'text': "Distribution Entropy"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, np.log(10)]}}
            ))
            st.plotly_chart(fig_entropy)

            cols = st.columns(2)
            cols[0].metric("Skewness", f"{score_dist['skewness']:.3f}")
            cols[1].metric("Kurtosis", f"{score_dist['kurtosis']:.3f}")

            if abs(score_dist['skewness']) > 1:
                st.warning("Score distribution is skewed. Consider adjusting weights.")
            if score_dist['kurtosis'] > 3:
                st.warning("Many extreme values detected. Review clipping range.")

    with col2:
        st.subheader("Component Contributions")

        contributions = latest_stats['score_formula']['contributions']
        fig_contrib = px.pie(
            values=list(contributions.values()),
            names=list(contributions.keys()),
            title="Score Component Contributions"
        )
        st.plotly_chart(fig_contrib)

        total_contrib = sum(contributions.values())
        if any(v / total_contrib > 0.6 for v in contributions.values()):
            st.warning("Component contribution is too high. Review weight balance.")

    st.subheader("Component Correlations")
    correlations = latest_stats['score_formula']['correlations']
    fig_corr = go.Figure(data=[
        go.Bar(
            x=list(correlations.keys()),
            y=list(correlations.values()),
            text=[f'{val:.3f}' for val in correlations.values()],
            textposition='auto'
        )
    ])
    fig_corr.update_layout(title="Component Correlation Coefficients")
    st.plotly_chart(fig_corr)

    st.subheader("Reaction-Sentiment Consistency")
    consistency = latest_stats['score_formula']['consistency']

    fig_consist = go.Figure(data=[
        go.Bar(
			x=list(consistency.keys()),
            y=list(consistency.values()),
            text=[f'{val:.3f}' for val in consistency.values()],
            textposition='auto'
        )
    ])
    fig_consist.update_layout(title="Consistency Metrics")
    st.plotly_chart(fig_consist)

    st.subheader("Improvement Suggestions")

    improvements = []

    if score_dist['entropy'] < np.log(5):
        improvements.append("Score distribution is too concentrated. Consider adjusting component weights to broaden distribution.")

    if any(abs(v) > 0.8 for v in correlations.values()):
        improvements.append("Some components show high correlation. Consider removing redundant signals.")

    if consistency['rate'] < 0.3:
        improvements.append("Low reaction-sentiment consistency. Consider increasing consistency bonus score.")

    for imp in improvements:
        st.info(imp)

    if not improvements:
        st.success("Current scoring formula is working well!")

if __name__ == "__main__":
    main()