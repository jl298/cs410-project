import numpy as np
import joblib
import json
import os
from datetime import datetime


class RecommenderInference:
    def __init__(self, model_dir="../model/recommender_model"):
        try:
            print(f"Loading model from {model_dir}")

            model_files = [f for f in os.listdir(model_dir) if f.startswith('model_') and f.endswith('.pkl')]
            if not model_files:
                raise ValueError("No model files found")

            latest_model = max(model_files)
            model_path = os.path.join(model_dir, latest_model)

            scaler_files = [f for f in os.listdir(model_dir) if f.startswith('scaler_') and f.endswith('.pkl')]
            if not scaler_files:
                raise ValueError("No scalar files found")
            latest_scalar = max(scaler_files)
            scaler_path = os.path.join(model_dir, latest_scalar)

            self.model_components = joblib.load(model_path)
            self.matrix_scaler = joblib.load(scaler_path)

            self.U = self.model_components['U']
            self.sigma = self.model_components['sigma']
            self.Vt = self.model_components['Vt']
            self.user_mapping = self.model_components['user_mapping']
            self.product_mapping = self.model_components['product_mapping']
            self.reverse_product_mapping = {v: k for k, v in self.product_mapping.items()}

            print(f"Model loaded successfully")
            print(f"Model dimensions - U: {self.U.shape}, sigma: {self.sigma.shape}, Vt: {self.Vt.shape}")
            print(f"Number of products in model: {len(self.product_mapping)}")

            catalog_path = os.path.join(model_dir, 'product_catalog.json')
            self.product_catalog = {}
            if os.path.exists(catalog_path):
                with open(catalog_path, 'r') as f:
                    self.product_catalog = json.load(f)
                print(f"Loaded product catalog with {len(self.product_catalog)} products")

        except Exception as e:
            raise Exception(f"Error initializing recommender: {str(e)}")

    def _validate_matrix_dimensions(self):
        n_features = self.U.shape[1]
        n_products = len(self.product_mapping)

        if self.Vt.shape[1] != n_products:
            print(f"Warning: Vt shape ({self.Vt.shape}) doesn't match product mapping size ({n_products})")

            if self.Vt.shape[1] < n_products:
                padding = np.zeros((self.Vt.shape[0], n_products - self.Vt.shape[1]))
                self.Vt = np.hstack((self.Vt, padding))
            else:
                self.Vt = self.Vt[:, :n_products]

        if len(self.sigma) != n_features:
            print(f"Warning: sigma length ({len(self.sigma)}) doesn't match feature count ({n_features})")

            if len(self.sigma) < n_features:
                padding = np.zeros(n_features - len(self.sigma))
                self.sigma = np.concatenate((self.sigma, padding))
            else:
                self.sigma = self.sigma[:n_features]

    def _get_popular_products(self, n=5):
        product_scores = np.mean(self.Vt, axis=0)
        top_indices = product_scores.argsort()[-n:][::-1]

        return [
            (self.reverse_product_mapping[idx], float(product_scores[idx]))
            for idx in top_indices
        ]

    def _convert_scores(self, sentiment_label: str, reaction: str) -> float:
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

            REACTION_WEIGHT = 0.7
            SENTIMENT_WEIGHT = 0.3

            sentiment_score = sentiment_map.get(sentiment_label, 0.0)
            reaction_score = reaction_map.get(reaction, 0.0)

            consistency_bonus = 0.1 if (
                    (reaction == 'liked' and sentiment_label == 'good') or
                    (reaction == 'ignored' and sentiment_label == 'bad')
            ) else 0.0

            score = (
                    reaction_score * REACTION_WEIGHT +
                    sentiment_score * SENTIMENT_WEIGHT +
                    consistency_bonus
            )

            return min(max(score, 0.0), 1.0)

        except Exception as e:
            print(f"Error in score conversion: {str(e)}")
            return 0.0

    def get_similar_products(self, product_id):
        if product_id not in self.product_mapping:
            print(f"Product {product_id} not in training set")
            return []

        product_idx = self.product_mapping[product_id]
        product_vector = self.Vt[:, product_idx]

        similarities = np.dot(self.Vt.T, product_vector)
        similar_indices = similarities.argsort()[-6:][::-1]

        similar_indices = similar_indices[similar_indices != product_idx]

        return [
            {
                'product_id': self.reverse_product_mapping[idx],
                'similarity': float(similarities[idx])
            }
            for idx in similar_indices[:5]
        ]

    def get_top_n_recommendations(self, user_vector, n=5, exclude_rated=True):
        try:
            user_latent = np.dot(user_vector, self.Vt.T)
            user_pred = np.dot(user_latent, self.Vt)
            user_pred = self.matrix_scaler.transform(user_pred.reshape(1, -1)).flatten()

            if exclude_rated:
                rated_items = np.where(user_vector > 0)[0]
                user_pred[rated_items] = -np.inf

            top_n_idx = user_pred.argsort()[-n:][::-1]

            recommendations = []
            for idx in top_n_idx:
                if user_pred[idx] != -np.inf and idx in self.reverse_product_mapping:
                    product_id = self.reverse_product_mapping[idx]
                    confidence = float(user_pred[idx])
                    recommendations.append({
                        'product_id': product_id,
                        'confidence': confidence,
                        'recommendation_type': 'matrix_factorization'
                    })

            return recommendations

        except Exception as e:
            print(f"Error in get_top_n_recommendations: {str(e)}")
            return []

    def recommend_for_user(self, user_data, n_recommendations=5):
        try:
            user_vector = np.zeros(self.Vt.shape[1])
            rated_products = []
            unknown_products = []

            for item in user_data:
                product_id = item['product_id']
                if product_id in self.product_mapping:
                    score = self._convert_scores(
                        item['sentiment_label'],
                        item['reaction']
                    )
                    user_vector[self.product_mapping[product_id]] = score
                    rated_products.append(product_id)
                else:
                    unknown_products.append(product_id)

            print(f"User has rated {len(rated_products)} known products: {rated_products}")
            if unknown_products:
                print(f"Found {len(unknown_products)} unknown products: {unknown_products}")

            recommendations = []

            if rated_products:
                try:
                    mf_recommendations = self.get_top_n_recommendations(user_vector, n=n_recommendations)
                    recommendations.extend(mf_recommendations)
                except Exception as e:
                    print(f"Matrix Factorization recommendations failed: {e}")

            if len(recommendations) < n_recommendations and rated_products:
                for product_id in rated_products:
                    similar_products = self.get_similar_products(product_id)
                    for prod in similar_products:
                        if len(recommendations) < n_recommendations and not any(r['product_id'] == prod['product_id'] for r in recommendations):
                            recommendations.append({
                                'product_id': prod['product_id'],
                                'confidence': prod['similarity'],
                                'recommendation_type': 'similarity'
                            })

            if not recommendations:
                print("Using popularity-based recommendations")
                popular_products = self._get_popular_products(n_recommendations)
                recommendations.extend([
                    {'product_id': pid, 'confidence': score, 'recommendation_type': 'popularity'}
                    for pid, score in popular_products
                ])

            print(f"Generated {len(recommendations)} total recommendations")
            return recommendations

        except Exception as e:
            print(f"Error in recommend_for_user: {str(e)}")
            return []


def load_json_data(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        online_data = [
            entry for entry in data
            if entry.get('tag') == 'online'
        ]

        if not online_data:
            return []

        online_data.sort(
            key=lambda x: datetime.fromisoformat(x['timestamp_infer']),
            reverse=True
        )

        return [online_data[0]]
    except Exception as e:
        raise Exception(f"Error loading JSON data: {str(e)}")


def main():
    try:
        recommender = RecommenderInference()
        latest_sentiment = load_json_data('../dataset/sentiment-dataset.json')

        if not latest_sentiment:
            print("No valid sentiment data found")
            return

        user_id = latest_sentiment[0]['user_id']
        print(f"\nGenerating recommendations for user {user_id}")
        recommendations = recommender.recommend_for_user(latest_sentiment)

        if recommendations:
            output_data = []
            source_product = latest_sentiment[0]['product_id']
            current_time = datetime.now().isoformat()

            for rec in recommendations:
                recommendation = {
                    'user_id': user_id,
                    'source_product': source_product,
                    'recommended_product': rec['product_id'],
                    'confidence': rec['confidence'],
                    'recommendation_type': rec.get('recommendation_type', 'matrix_factorization'),
                    'timestamp': current_time
                }
                output_data.append(recommendation)

                print(f"Recommendation: {recommendation}")

            output_path = '../dataset/recommendations-dataset.json'
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nRecommendations saved to {output_path}")
        else:
            print(f"No recommendations generated for user {user_id}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()