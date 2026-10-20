"""
train_and_evaluate.py - Training, Metrics & Probability Analysis

Demonstrates:
- Stratified Train/Test Split
- Naive Bayes Training
- Evaluation Metrics: Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Top Spam vs. Ham Feature Weights (Odds Ratios)
"""

import csv
import json
import math
import random
from typing import List, Tuple
from src.text_processor import TextProcessor
from src.naive_bayes import NaiveBayesClassifier


def load_dataset(csv_path: str) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def stratified_split(
    texts: List[str], labels: List[str], test_ratio: float = 0.2, seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Splits data preserving the class balance."""
    random.seed(seed)
    spam_samples = [(t, l) for t, l in zip(texts, labels) if l == "spam"]
    ham_samples = [(t, l) for t, l in zip(texts, labels) if l == "ham"]

    random.shuffle(spam_samples)
    random.shuffle(ham_samples)

    n_test_spam = max(1, int(len(spam_samples) * test_ratio))
    n_test_ham = max(1, int(len(ham_samples) * test_ratio))

    test_pairs = spam_samples[:n_test_spam] + ham_samples[:n_test_ham]
    train_pairs = spam_samples[n_test_spam:] + ham_samples[n_test_ham:]

    random.shuffle(train_pairs)
    random.shuffle(test_pairs)

    x_train, y_train = [t for t, _ in train_pairs], [l for _, l in train_pairs]
    x_test, y_test = [t for t, _ in test_pairs], [l for _, l in test_pairs]

    return x_train, y_train, x_test, y_test


def evaluate_classifier(clf: NaiveBayesClassifier, x_test: List[str], y_test: List[str]):
    tp = 0  # Actual Spam, Predicted Spam
    fp = 0  # Actual Ham, Predicted Spam (False Alarm)
    tn = 0  # Actual Ham, Predicted Ham (Correctly allowed)
    fn = 0  # Actual Spam, Predicted Ham (Missed Spam)

    print("\n" + "=" * 65)
    print(" 🧪 TEST SET EVALUATION DETAILS")
    print("=" * 65)

    for i, (text, actual) in enumerate(zip(x_test, y_test), 1):
        pred_label = clf.predict(text)
        proba = clf.predict_proba(text)
        pred_normalized = "spam" if pred_label == "Spam" else "ham"

        is_correct = (pred_normalized == actual)
        status_icon = "✅" if is_correct else "❌"

        if actual == "spam" and pred_normalized == "spam":
            tp += 1
        elif actual == "ham" and pred_normalized == "spam":
            fp += 1
        elif actual == "ham" and pred_normalized == "ham":
            tn += 1
        elif actual == "spam" and pred_normalized == "ham":
            fn += 1

        print(f"[{i:02d}] {status_icon} Actual: {actual.upper():<4} | Predicted: {pred_normalized.upper():<4} (Spam Prob: {proba['spam'] * 100:5.1f}%)")
        print(f"     Preview: {text[:60]}...")

    total = len(y_test)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 65)
    print(" 📊 PERFORMANCE METRICS")
    print("=" * 65)
    print(f"Total Test Samples: {total}")
    print(f"Accuracy  : {accuracy * 100:.2f}%  (Overall correct classifications)")
    print(f"Precision : {precision * 100:.2f}%  (When predicted as Spam, how often was it real spam?)")
    print(f"Recall    : {recall * 100:.2f}%  (Of all real spam emails, what fraction did we catch?)")
    print(f"F1-Score  : {f1:.4f}   (Harmonic mean of precision and recall)")

    print("\n" + "-" * 65)
    print(" 🧮 CONFUSION MATRIX")
    print("-" * 65)
    print(f"                 Predicted Spam    Predicted Not Spam")
    print(f"  Actual Spam      TP = {tp:<6}        FN = {fn:<6}")
    print(f"  Actual Not Spam  FP = {fp:<6}        TN = {tn:<6}")
    print("-" * 65)


def print_probability_insights(clf: NaiveBayesClassifier):
    print("\n" + "=" * 65)
    print(" 📐 PROBABILITY INSIGHTS (LEARNED PARAMETERS)")
    print("=" * 65)
    p_spam = math.exp(clf.log_priors["spam"])
    p_ham = math.exp(clf.log_priors["ham"])
    print(f"• Prior P(Spam)     : {p_spam:.4f} ({p_spam * 100:.1f}%)")
    print(f"• Prior P(Not Spam) : {p_ham:.4f} ({p_ham * 100:.1f}%)")
    print(f"• Total Vocabulary  : {len(clf.vocab)} unique words")
    print(f"• Total Words in Spam Corpus : {clf.total_words['spam']}")
    print(f"• Total Words in Ham Corpus  : {clf.total_words['ham']}")

    top_indicators = clf.top_indicator_words(top_n=8)

    print("\n🔥 Top Words that Indicate SPAM (High Likelihood Ratio):")
    print(f"  {'Word':<15} {'P(w|Spam)':<14} {'P(w|Ham)':<14} {'Spam Odds Multiplier'}")
    print("  " + "-" * 60)
    for word, ratio in top_indicators["top_spam_words"]:
        p_s = clf.word_likelihood(word, "spam")
        p_h = clf.word_likelihood(word, "ham")
        print(f"  {word:<15} {p_s:<14.5f} {p_h:<14.5f} {ratio:>6.2f}x more likely in spam")

    print("\n💼 Top Words that Indicate NOT SPAM (Ham):")
    print(f"  {'Word':<15} {'P(w|Ham)':<14} {'P(w|Spam)':<14} {'Ham Odds Multiplier'}")
    print("  " + "-" * 60)
    for word, ratio in top_indicators["top_ham_words"]:
        p_s = clf.word_likelihood(word, "spam")
        p_h = clf.word_likelihood(word, "ham")
        print(f"  {word:<15} {p_h:<14.5f} {p_s:<14.5f} {ratio:>6.2f}x more likely in ham")
    print("=" * 65)


def main():
    csv_path = "data/emails.csv"
    texts, labels = load_dataset(csv_path)
    print(f"Loaded {len(texts)} emails from {csv_path}.")

    x_train, y_train, x_test, y_test = stratified_split(texts, labels, test_ratio=0.2, seed=42)
    print(f"Split dataset: {len(x_train)} training emails, {len(x_test)} testing emails.")

    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(x_train, y_train)

    print_probability_insights(clf)
    evaluate_classifier(clf, x_test, y_test)


if __name__ == "__main__":
    main()
