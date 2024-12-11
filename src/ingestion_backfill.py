import json
import hashlib
import random
import logging
from collections import defaultdict
import copy
import numpy as np
from typing import Dict, List
import itertools
from functools import lru_cache

MAX_POSTS = 100
MAX_COMMENTS_PER_POST = 5
DEFAULT_COMMENT_TEXT = "No comment text available"
DEFAULT_USER_NAME = "Unknown User"

SENTIMENT_RATIO = 0.35
RECOMMENDER_RATIO = 0.65

INPUT_PATH_REVIEW = '../dataset/Video_Games_demo.json'
OUTPUT_PATH_USER = '../dataset/sns-user-dataset.json'
OUTPUT_PATH_POSTS = '../dataset/sns-posts-dataset.json'
OUTPUT_PATH_SENTIMENT = '../dataset/sns-posts-dataset.sentiment.json'
OUTPUT_PATH_RECOMMENDER = '../dataset/sns-posts-dataset.recommender.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@lru_cache(maxsize=1024)
def hash_asin(asin: str) -> str:
    try:
        hash_object = hashlib.md5(str(asin).encode())
        return f"user_{hash_object.hexdigest()[:8]}"
    except Exception as e:
        logging.error(f"Error hashing asin {asin}: {str(e)}")
        return f"user_invalid"

REACTION_MAP = {
    0: "ignored",
    1: "viewed",
    2: "liked"
}

def get_reaction(overall: float) -> str:
    try:
        overall = float(overall)
        if overall >= 4.0:
            return "liked"
        elif overall == 3.0:
            return "viewed"
        return "ignored"
    except (ValueError, TypeError):
        return "ignored"

def is_valid_review(review: dict) -> bool:
    return all(field in review for field in ('asin', 'reviewerID'))

def safe_get(dict_obj: dict, key: str, default: str = "") -> str:
    try:
        return dict_obj.get(key, default)
    except Exception as e:
        logging.warning(f"Error accessing key {key}: {str(e)}")
        return default


def split_posts_by_rating(posts: List[dict]) -> tuple[List[dict], List[dict]]:
    posts_by_rating = defaultdict(list)

    for post in posts:
        total_score = 0
        valid_reviews = 0

        for comment in post['comments']:
            if comment['reaction'] == 'liked':
                total_score += 5
            elif comment['reaction'] == 'viewed':
                total_score += 3
            else:
                total_score += 1
            valid_reviews += 1

        if valid_reviews > 0:
            avg_score = total_score / valid_reviews
            posts_by_rating[round(avg_score)].append(post)

    min_samples_per_class = 2

    for score in range(1, 6):
        if len(posts_by_rating[score]) < min_samples_per_class:
            needed_samples = min_samples_per_class - len(posts_by_rating[score])
            other_posts = [p for s, posts in posts_by_rating.items()
                           if s != score for p in posts]

            if other_posts:
                additional_posts = random.choices(other_posts, k=needed_samples)

                for post in additional_posts:
                    new_post = copy.deepcopy(post)
                    for comment in new_post['comments']:
                        if score <= 2:
                            comment['reaction'] = 'ignored'
                        elif score == 3:
                            comment['reaction'] = 'viewed'
                        else:
                            comment['reaction'] = 'liked'

                    posts_by_rating[score].append(new_post)

    sentiment_posts = []
    recommender_posts = []

    for score, score_posts in posts_by_rating.items():
        n_sentiment = max(int(len(score_posts) * SENTIMENT_RATIO), min_samples_per_class)

        score_posts.sort(key=lambda x: min(comment['timestamp_comment']
                                           for comment in x['comments']))

        sentiment_posts.extend(score_posts[:n_sentiment])
        recommender_posts.extend(score_posts[n_sentiment:])

    print(f"\nData split summary:")
    print(f"Total posts: {len(posts)}")
    print(f"Sentiment dataset: {len(sentiment_posts)} posts")
    print(f"Recommender dataset: {len(recommender_posts)} posts")

    print("\nSentiment dataset class distribution:")
    sentiment_dist = defaultdict(int)
    for post in sentiment_posts:
        for comment in post['comments']:
            sentiment_dist[comment['reaction']] += 1

    for reaction, count in sentiment_dist.items():
        print(f"{reaction}: {count}")

    return sentiment_posts, recommender_posts

def process_reviews() -> None:
    reviews_by_asin: Dict[str, List[dict]] = defaultdict(list)
    reviews_by_user: Dict[str, List[dict]] = defaultdict(list)

    try:
        BATCH_SIZE = 10000
        with open(INPUT_PATH_REVIEW, 'r', encoding='utf-8') as f:
            for batch_num, lines in enumerate(itertools.zip_longest(*[f] * BATCH_SIZE)):
                lines = [line for line in lines if line is not None]
                for line_num, line in enumerate(lines, start=batch_num * BATCH_SIZE + 1):
                    try:
                        review = json.loads(line.strip())
                        if not is_valid_review(review):
                            continue

                        reviews_by_asin[review['asin']].append(review)
                        reviews_by_user[review['reviewerID']].append(review)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON decode error at line {line_num}: {str(e)}")
                        continue
                    except Exception as e:
                        logging.error(f"Error processing line {line_num}: {str(e)}")
                        continue

        if not reviews_by_asin:
            raise ValueError("No valid reviews found in input file")

        posts = []
        asin_items = list(reviews_by_asin.items())[:MAX_POSTS]
        logging.info(f"Processing {len(asin_items)} posts (limited from {len(reviews_by_asin)})")

        for post_id, (asin, reviews) in enumerate(asin_items):
            comments = []
            reaction_counts = {'liked': 0, 'viewed': 0, 'ignored': 0}

            limited_reviews = reviews[:MAX_COMMENTS_PER_POST]
            if len(reviews) > MAX_COMMENTS_PER_POST:
                logging.info(f"Limiting comments for post {post_id} from {len(reviews)} to {MAX_COMMENTS_PER_POST}")

            for comment_id, review in enumerate(limited_reviews):
                reaction = get_reaction(safe_get(review, 'overall', 0))
                reaction_counts[reaction] += 1

                comment = {
                    "comment_id": str(comment_id),
                    "user_id": safe_get(review, 'reviewerID', 'unknown_user'),
                    "comment_text": " ".join([
                        safe_get(review, 'summary', DEFAULT_COMMENT_TEXT),
                        safe_get(review, 'reviewText', DEFAULT_COMMENT_TEXT)
                    ]),
                    "timestamp_comment": safe_get(review, 'unixReviewTime', 0),
                    "reaction": reaction,
                    "tag": "offline"
                }
                comments.append(comment)

            post = {
                "post_id": str(post_id),
                "user_id": hash_asin(asin),
                "product_id": asin,
                "post_text": "Click here to view product information.",
                "like_count": reaction_counts['liked'],
                "view_count": reaction_counts['viewed'],
                "ignore_count": reaction_counts['ignored'],
                "click_count": 0,
                "comments": comments,
                "tag": "offline"
            }
            posts.append(post)

        with open(OUTPUT_PATH_POSTS, 'w', encoding='utf-8') as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully wrote {len(posts)} items to {OUTPUT_PATH_POSTS}")

        sentiment_posts, recommender_posts = split_posts_by_rating(posts)

        with open(OUTPUT_PATH_SENTIMENT, 'w', encoding='utf-8') as f:
            json.dump(sentiment_posts, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully wrote {len(sentiment_posts)} items to {OUTPUT_PATH_SENTIMENT}")

        with open(OUTPUT_PATH_RECOMMENDER, 'w', encoding='utf-8') as f:
            json.dump(recommender_posts, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully wrote {len(recommender_posts)} items to {OUTPUT_PATH_RECOMMENDER}")

        random_user_id = random.choice(list(reviews_by_user.keys()))
        generate_user_json({random_user_id: reviews_by_user[random_user_id]})

    except FileNotFoundError:
        logging.error(f"Input file {INPUT_PATH_REVIEW} not found")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in process_reviews: {str(e)}")
        raise

def generate_user_json(reviews_by_user: Dict[str, List[dict]]) -> None:
    try:
        random_user_id = random.choice(list(reviews_by_user.keys()))
        user_reviews = reviews_by_user[random_user_id]

        user_name = safe_get(user_reviews[0], 'reviewerName', DEFAULT_USER_NAME)

        reaction_lists = defaultdict(list)

        for review in user_reviews:
            product_id = safe_get(review, 'asin')
            if not product_id:
                continue

            reaction = get_reaction(safe_get(review, 'overall', 0))
            reaction_lists[reaction].append(product_id)

        user_data = {
            "user_id": random_user_id,
            "user_name": user_name,
            "liked_list": reaction_lists["liked"],
            "viewed_list": reaction_lists["viewed"],
            "ignored_list": reaction_lists["ignored"],
            "clicked_list": [],
        }

        with open(OUTPUT_PATH_USER, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully wrote {OUTPUT_PATH_USER}")

    except Exception as e:
        logging.error(f"Error in generate_user_json: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        process_reviews()
    except Exception as e:
        logging.error(f"Script failed: {str(e)}")
        exit(1)