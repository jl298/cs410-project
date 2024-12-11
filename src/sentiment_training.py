import json
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

BATCH_SIZE = 16

INPUT_PATH_SENTIMENT = '../dataset/sns-posts-dataset.sentiment.json'

def load_sns_posts(file_path):
    with open(file_path, 'r') as f:
        posts = json.load(f)

    comments_data = []
    for post in posts:
        for comment in post['comments']:
            if 'tag' in comment and comment['tag'] == 'offline':
                if comment['reaction'] == 'liked':
                    label = 2
                elif comment['reaction'] == 'viewed':
                    label = 1
                else:
                    label = 0

                comments_data.append({
                    'text': comment['comment_text'],
                    'label': label
                })

    df = pd.DataFrame(comments_data)
    print(f"Loaded {len(df)} valid offline comments")
    print("\nLabel distribution:")
    print(df['label'].value_counts().sort_index())
    print("\nSample data:")
    print(df.head())
    return df


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )


def prepare_dataset(df, tokenizer):
    dataset = Dataset.from_pandas(df)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    tokenized_dataset = tokenized_dataset.add_column('labels', df['label'])
    return tokenized_dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    report = classification_report(
        labels,
        preds,
        labels=[0, 1, 2],
        target_names=['ignored', 'viewed', 'liked'],
        output_dict=True
    )

    return {
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'ignored_f1': report['ignored']['f1-score'],
        'viewed_f1': report['viewed']['f1-score'],
        'liked_f1': report['liked']['f1-score']
    }

def train_sentiment_model(df, model_name="distilbert-base-uncased"):
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df['label'],
        random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    best_model = None
    best_val_score = float('-inf')

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
        print(f"\nTraining fold {fold + 1}/5...")

        fold_train_df = train_val_df.iloc[train_idx]
        fold_val_df = train_val_df.iloc[val_idx]

        train_dataset = prepare_dataset(fold_train_df, tokenizer)
        val_dataset = prepare_dataset(fold_val_df, tokenizer)

        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3
        )

        training_args = TrainingArguments(
            output_dir=f"../training/sns_sentiment_results/fold-{fold}",
            num_train_epochs=3,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"../training/sns_sentiment_logs/fold-{fold}",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        eval_results = trainer.evaluate()
        cv_scores.append(eval_results['eval_macro_f1'])

        if eval_results['eval_macro_f1'] > best_val_score:
            best_val_score = eval_results['eval_macro_f1']
            best_model = trainer.model

        print_fold_results(fold, eval_results)

    print_cv_results(cv_scores)

    test_dataset = prepare_dataset(test_df, tokenizer)
    final_trainer = Trainer(
        model=best_model,
        compute_metrics=compute_metrics
    )

    test_results = final_trainer.evaluate(test_dataset)
    print("\nFinal Test Results:")
    print_test_results(test_results)

    print("\nTraining final model on full training data...")
    full_train_dataset = prepare_dataset(train_val_df, tokenizer)

    final_model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    final_training_args = TrainingArguments(
        output_dir="../training/sns_sentiment_results/final",
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="../training/sns_sentiment_logs/final",
        logging_steps=10,
        save_strategy="epoch"
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_training_args,
        train_dataset=full_train_dataset
    )

    final_trainer.train()

    return final_trainer, tokenizer, test_results

def print_fold_results(fold, results):
    print(f"Fold {fold + 1} Results:")
    print(f"Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Macro F1: {results['eval_macro_f1']:.4f}")
    print(f"Ignored F1: {results['eval_ignored_f1']:.4f}")
    print(f"Viewed F1: {results['eval_viewed_f1']:.4f}")
    print(f"Liked F1: {results['eval_liked_f1']:.4f}")

def print_cv_results(cv_scores):
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print("\nCross-validation results:")
    print(f"Mean F1: {cv_mean:.3f} (+/- {cv_std * 2:.3f})")

def print_test_results(results):
    print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Test Macro F1: {results['eval_macro_f1']:.4f}")
    print(f"Test Ignored F1: {results['eval_ignored_f1']:.4f}")
    print(f"Test Viewed F1: {results['eval_viewed_f1']:.4f}")
    print(f"Test Liked F1: {results['eval_liked_f1']:.4f}")

def main():
    print("Loading SNS posts...")
    df = load_sns_posts(INPUT_PATH_SENTIMENT)

    print("\nStarting model training with validation...")
    trainer, tokenizer, test_results = train_sentiment_model(df)

    print("\nSaving model and tokenizer...")
    trainer.save_model("../model/sns_sentiment_model")
    tokenizer.save_pretrained("../model/sns_sentiment_model")

    with open("../model/sns_sentiment_model/test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()