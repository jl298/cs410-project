import json
import torch
import argparse
from datetime import datetime
from pathlib import Path
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np

MODEL_PATH = "../model/sns_sentiment_model"
INPUT_USER_OFFLINE_DATASET = "../dataset/sns-posts-dataset.recommender.json"
INPUT_USER_ONLINE_DATASET = "../dataset/sns-posts-dataset.json"
OUTPUT_SENTIMENT_DATASET = "../dataset/sentiment-dataset.json"
BATCH_SIZE = 16
MAX_LENGTH = 512

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_model_and_tokenizer(model_path: str) -> Tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        return model, tokenizer, device
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {str(e)}")
        raise

def load_test_metrics(model_path: str) -> Dict:
    metrics_path = Path(model_path) / "test_results.json"
    try:
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logging.warning(f"Could not load test metrics: {str(e)}")
        return {}


def load_user_data(user_file_path: str) -> dict:
    try:
        with open(user_file_path, 'r') as f:
            data = json.load(f)

            if isinstance(data, list):
                return {
                    'user_id': None,
                    'post_id_of_last_comment': None,
                    'last_comment_id': None
                }

            return {
                'user_id': data.get('user_id'),
                'post_id_of_last_comment': data.get('post_id_of_last_comment'),
                'last_comment_id': data.get('last_comment_id')
            }
    except Exception as e:
        logging.error(f"Error loading user data from {user_file_path}: {str(e)}")
        return {
            'user_id': None,
            'post_id_of_last_comment': None,
            'last_comment_id': None
        }


def load_posts(posts_file_path: str, post_id_of_last_comment: str = None, last_comment_id: str = None,
               exclude_last: bool = False) -> List[Dict]:
    try:
        with open(posts_file_path, 'r') as f:
            data = json.load(f)

        posts = data if isinstance(data, list) else data.get('posts', [])

        if not isinstance(posts, list):
            raise ValueError("Expected posts data to be a list of posts")

        result = []

        if exclude_last:
            try:
                user_data = load_user_data(INPUT_USER_ONLINE_DATASET)
                last_post_id = user_data.get('post_id_of_last_comment')
                last_comm_id = user_data.get('last_comment_id')
            except Exception as e:
                logging.warning(f"Could not load last comment info: {str(e)}")
                last_post_id = None
                last_comm_id = None

            for post in posts:
                for comment in post['comments']:
                    if not (last_post_id and last_comm_id and
                            post['post_id'] == last_post_id and
                            comment['comment_id'] == last_comm_id):
                        result.append({
                            'user_id': comment['user_id'],
                            'product_id': post['product_id'],
                            'comment_text': comment['comment_text'],
                            'reaction': comment['reaction'],
                            'post_id': post['post_id'],
                            'comment_id': comment['comment_id']
                        })

        elif post_id_of_last_comment is not None and last_comment_id is not None:
            for post in posts:
                if post['post_id'] == post_id_of_last_comment:
                    for comment in post['comments']:
                        if comment['comment_id'] == last_comment_id:
                            result.append({
                                'user_id': comment['user_id'],
                                'product_id': post['product_id'],
                                'comment_text': comment['comment_text'],
                                'reaction': comment['reaction'],
                                'post_id': post['post_id'],
                                'comment_id': comment['comment_id']
                            })
                            break

        else:
            for post in posts:
                for comment in post['comments']:
                    if comment.get('tag') == 'online':
                        result.append({
                            'user_id': comment['user_id'],
                            'product_id': post['product_id'],
                            'comment_text': comment['comment_text'],
                            'reaction': comment['reaction'],
                            'post_id': post['post_id'],
                            'comment_id': comment['comment_id']
                        })

        logging.info(f"Found {len(result)} matching comments")
        return result

    except Exception as e:
        logging.error(f"Error loading posts from {posts_file_path}: {str(e)}")
        return []

def convert_label_to_sentiment(label: int, probs: np.ndarray) -> Tuple[str, float]:
    sentiment_map = {
        2: "good",
        1: "moderate",
        0: "bad"
    }
    confidence = float(probs[label])
    return sentiment_map.get(label, "bad"), confidence


def load_existing_sentiments(file_path: str) -> List[Dict]:
    try:
        if Path(file_path).exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logging.error(f"Error loading existing sentiments: {str(e)}")
        return []


def save_sentiments(results: List[Dict], file_path: str, append: bool = True) -> None:
    try:
        if append:
            existing_data = load_existing_sentiments(file_path)
            results = existing_data + results

        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {file_path}")

    except Exception as e:
        logging.error(f"Error saving results: {str(e)}")


def create_batches(comments: List[Dict], batch_size: int):
    logging.info(f"Creating batches from {len(comments)} comments with batch size {batch_size}")
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i + batch_size]
        texts = [c['comment_text'] for c in batch]
        yield batch, texts


def inference_sentiment(
        model_path: str,
        comments_data: List[Dict],
        user_id: Optional[str] = None,
        is_backfill: bool = False
) -> List[Dict]:
    try:
        logging.info(f"Starting inference with {len(comments_data)} comments")

        model, tokenizer, device = load_model_and_tokenizer(model_path)
        test_metrics = load_test_metrics(model_path)
        if test_metrics:
            logging.info(f"Model test metrics: {test_metrics}")

        comments_to_process = comments_data
        if user_id:
            comments_to_process = [c for c in comments_data if c['user_id'] == user_id]
            logging.info(f"Filtered {len(comments_to_process)} comments for user {user_id}")

        if not comments_to_process:
            logging.warning("No comments to process")
            return []

        print(f"Processing {len(comments_to_process)} comments...")
        results = []

        with torch.no_grad():
            for batch_comments, batch_texts in create_batches(comments_to_process, BATCH_SIZE):
                print(f"Processing batch of {len(batch_texts)} comments")

                inputs = tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors='pt'
                )

                print(f"Tokenizer output shapes: {[(k, v.shape) for k, v in inputs.items()]}")

                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = logits.argmax(-1)

                predictions = predictions.cpu().numpy()
                probabilities = probabilities.cpu().numpy()

                print(f"Predictions shape: {predictions.shape}, values: {predictions}")

                batch_results = []
                for comment, pred, probs in zip(batch_comments, predictions, probabilities):
                    sentiment, confidence = convert_label_to_sentiment(int(pred), probs)
                    result = {
                        'user_id': comment['user_id'],
                        'product_id': comment['product_id'],
                        'sentiment_label': sentiment,
                        'confidence': confidence,
                        'reaction': comment['reaction'],
                        'tag': 'offline' if is_backfill else 'online',
                        'timestamp_infer': datetime.now().isoformat()
                    }
                    batch_results.append(result)
                    print(f"Created result: {result}")

                results.extend(batch_results)
                logging.info(f"Processed batch of {len(batch_results)} comments")

            logging.info(f"Completed inference for {len(results)} comments")
            return results
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backfill', action='store_true',
                        help='Process all comments if true, otherwise process only user comments')
    args = parser.parse_args()

    try:
        if args.backfill:
            logging.info("Running backfill for all comments except last...")
            posts = load_posts(INPUT_USER_OFFLINE_DATASET, exclude_last=True)
            results = inference_sentiment(MODEL_PATH, posts, is_backfill=True)
            if results:
                save_sentiments(results, OUTPUT_SENTIMENT_DATASET, append=False)
        else:
            logging.info("Processing single user comments...")
            user_data = load_user_data(INPUT_USER_ONLINE_DATASET)
            user_id = user_data['user_id']
            post_id_of_last_comment = user_data['post_id_of_last_comment']
            last_comment_id = user_data['last_comment_id']

            posts = load_posts(INPUT_USER_ONLINE_DATASET, post_id_of_last_comment, last_comment_id)
            if not posts:
                logging.warning("No matching comment found")
                return

            results = inference_sentiment(MODEL_PATH, posts, user_id, is_backfill=False)
            if results:
                save_sentiments(results, OUTPUT_SENTIMENT_DATASET, append=True)

        if results:
            logging.info("\nSentiment Analysis Results:")
            logging.info("user_id | product_id | sentiment | confidence | reaction")
            logging.info("-" * 70)
            for result in results[:5]:
                logging.info(
                    f"{result['user_id']} | {result['product_id']} | "
                    f"{result['sentiment_label']} | {result['confidence']:.3f} | {result['reaction']}"
                )
            logging.info(f"... and {len(results) - 5} more results")
        else:
            logging.warning("No results generated")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()