import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import os
import sys
import logging
from typing import Dict, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RecommenderTrainer:
    def __init__(self,
                 model_dir: str = "../model/recommender_model",
                 reaction_weight: float = 0.7,
                 sentiment_weight: float = 0.3):
        self.model_dir = model_dir
        self.matrix_scaler = MinMaxScaler()
        self.reaction_weight = reaction_weight
        self.sentiment_weight = sentiment_weight
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            df = pd.DataFrame(data)
            logging.info(f"Loaded {len(df)} interactions")

            return df

        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def _convert_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            sentiment_map = {
                'good': 1.0,
                'moderate': 0.5,
                'bad': 0.0
            }
            reaction_map = {
                'liked': 1.0,
                'viewed': 0.5,
                'ignored': 0.0
            }

            df['sentiment_score'] = df['sentiment_label'].map(sentiment_map)
            df['reaction_score'] = df['reaction'].map(reaction_map)

            consistency_bonus = 0.1
            df['consistency_score'] = df.apply(
                lambda row: consistency_bonus if (
                        (row['reaction'] == 'liked' and row['sentiment_label'] == 'good') or
                        (row['reaction'] == 'ignored' and row['sentiment_label'] == 'bad')
                ) else 0.0,
                axis=1
            )

            df['score'] = (
                    df['reaction_score'] * self.reaction_weight +
                    df['sentiment_score'] * self.sentiment_weight +
                    df['consistency_score']
            )

            df['score'] = df['score'].clip(0, 1)

            score_metrics = self.calculate_score_metrics(df)
            metrics_path = os.path.join(self.model_dir, f'score_metrics_{self.timestamp}.json')

            with open(metrics_path, 'w') as f:
                json.dump(score_metrics, f, indent=2)

            logging.info(f"Score metrics saved to {metrics_path}")

            return df

        except Exception as e:
            logging.error(f"Error in score conversion: {str(e)}")
            raise

    def calculate_score_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        from scipy import stats

        scores = df['score'].values
        reaction_scores = df['reaction_score'].values
        sentiment_scores = df['sentiment_score'].values
        consistency_scores = df['consistency_score'].values

        hist, _ = np.histogram(scores, bins=10, density=True)

        distribution_metrics = {
            'score_entropy': float(stats.entropy(hist + 1e-10)),
            'score_skewness': float(stats.skew(scores)),
            'score_kurtosis': float(stats.kurtosis(scores))
        }

        correlation_metrics = {
            'reaction_sentiment_corr': float(np.corrcoef(reaction_scores, sentiment_scores)[0, 1]),
            'reaction_consistency_corr': float(np.corrcoef(reaction_scores, consistency_scores)[0, 1]),
            'sentiment_consistency_corr': float(np.corrcoef(sentiment_scores, consistency_scores)[0, 1])
        }

        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            score_changes = np.diff(df_sorted['score'].values)
            temporal_metrics = {
                'temporal_consistency': float(np.std(score_changes)),
                'temporal_trend': float(np.polyfit(range(len(scores)), scores, 1)[0])
            }
        else:
            temporal_metrics = {}

        component_metrics = {
            'reaction_contribution': float(np.mean(reaction_scores * self.reaction_weight)),
            'sentiment_contribution': float(np.mean(sentiment_scores * self.sentiment_weight)),
            'consistency_contribution': float(np.mean(consistency_scores))
        }

        consistency_metrics = {
            'consistency_rate': float(np.mean(consistency_scores > 0)),
            'good_liked_rate': float(np.mean(
                (df['sentiment_label'] == 'good') & (df['reaction'] == 'liked')
            )),
            'bad_ignored_rate': float(np.mean(
                (df['sentiment_label'] == 'bad') & (df['reaction'] == 'ignored')
            ))
        }

        return {
            **distribution_metrics,
            **correlation_metrics,
            **temporal_metrics,
            **component_metrics,
            **consistency_metrics
        }

    def create_interaction_matrix(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict, Dict]:
        unique_users = df['user_id'].unique()
        unique_products = df['product_id'].unique()

        user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        product_mapping = {prod: idx for idx, prod in enumerate(unique_products)}

        matrix = np.zeros((len(unique_users), len(unique_products)))

        for _, row in df.iterrows():
            user_idx = user_mapping[row['user_id']]
            prod_idx = product_mapping[row['product_id']]
            matrix[user_idx, prod_idx] = row['score']

        return matrix, user_mapping, product_mapping

    def split_interaction_matrix(self, matrix: np.ndarray, test_size: float = 0.15
                                 ) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.random.rand(*matrix.shape) < test_size

        for i in range(matrix.shape[0]):
            nonzero_indices = matrix[i].nonzero()[0]
            if len(nonzero_indices) > 0 and mask[i, nonzero_indices].all():
                rand_idx = np.random.choice(nonzero_indices)
                mask[i, rand_idx] = False

        train = matrix.copy()
        test = matrix.copy()

        train[mask] = 0
        test[~mask] = 0

        return train, test

    def evaluate_recommendations(self, predicted: np.ndarray, actual: np.ndarray,
                                 k: int = 10) -> Dict[str, float]:
        mask = actual != 0
        rmse = np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))

        metrics = {
            'rmse': float(rmse),
            f'precision@{k}': 0.0,
            f'recall@{k}': 0.0,
            f'ndcg@{k}': 0.0
        }

        precision_k, recall_k, ndcg_k = [], [], []

        for user_idx in range(actual.shape[0]):
            actual_items = set(np.where(actual[user_idx] > 0)[0])
            if not actual_items:
                continue

            pred_scores = predicted[user_idx]
            top_k = np.argsort(pred_scores)[-k:][::-1]

            hits = len(set(top_k) & actual_items)
            precision_k.append(hits / k)
            recall_k.append(hits / len(actual_items))

            dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(top_k) if item in actual_items)
            idcg = sum(1 / np.log2(i + 2) for i in range(min(k, len(actual_items))))
            ndcg_k.append(dcg / idcg if idcg > 0 else 0)

        if precision_k:
            metrics[f'precision@{k}'] = float(np.mean(precision_k))
            metrics[f'recall@{k}'] = float(np.mean(recall_k))
            metrics[f'ndcg@{k}'] = float(np.mean(ndcg_k))

        return metrics

    def train(self, data_path: str, n_factors: int = 50, test_size: float = 0.15) -> Dict:
        try:
            df = self.load_data(data_path)
            df = self._convert_scores(df)

            matrix, user_mapping, product_mapping = self.create_interaction_matrix(df)

            train_matrix, test_matrix = self.split_interaction_matrix(matrix, test_size)

            k = min(n_factors, min(train_matrix.shape) - 1)
            U, sigma, Vt = svds(train_matrix, k=k)
            sigma = np.diag(sigma)

            predicted = np.dot(np.dot(U, sigma), Vt)
            predicted = self.matrix_scaler.fit_transform(predicted)
            metrics = self.evaluate_recommendations(predicted, test_matrix)

            final_U, final_sigma, final_Vt = svds(matrix, k=k)
            final_sigma = np.diag(final_sigma)

            model_components = {
                'U': final_U,
                'sigma': final_sigma,
                'Vt': final_Vt,
                'user_mapping': user_mapping,
                'product_mapping': product_mapping,
                'n_factors': k
            }

            os.makedirs(self.model_dir, exist_ok=True)

            for f in os.listdir(self.model_dir):
                if f.startswith('model_') or f.startswith('scaler_'):
                    os.remove(os.path.join(self.model_dir, f))

            model_path = os.path.join(self.model_dir, f'model_{self.timestamp}.pkl')
            scaler_path = os.path.join(self.model_dir, f'scaler_{self.timestamp}.pkl')
            metrics_path = os.path.join(self.model_dir, f'metrics_{self.timestamp}.json')

            joblib.dump(model_components, model_path)
            joblib.dump(self.matrix_scaler, scaler_path)

            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            logging.info(f"Model saved to {model_path}")
            logging.info(f"Scaler saved to {scaler_path}")
            logging.info(f"Training metrics: {metrics}")

            return metrics

        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise


def main():
    try:
        trainer = RecommenderTrainer(
            reaction_weight=0.7,
            sentiment_weight=0.3
        )

        metrics = trainer.train(
            "../dataset/sentiment-dataset.json",
            n_factors=50,
            test_size=0.15
        )

        print("\nTraining Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()