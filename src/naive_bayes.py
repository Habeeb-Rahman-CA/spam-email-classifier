"""
naive_bayes.py - Multinomial Naive Bayes Classifier from Scratch

Mathematical Foundations:
-------------------------
1. Bayes' Theorem:
       P(Class | Document) = P(Class) * P(Document | Class) / P(Document)

2. "Naive" Conditional Independence Assumption:
       P(w1, w2, ..., wn | Class) = ∏ P(wi | Class)

3. Laplace (Additive) Smoothing:
       P(w | Class) = (count(w, Class) + α) / (total_words_in_Class + α * |V|)
   Prevents zero-probability problem for words unseen in training.

4. Log-Probabilities (Numerical Underflow Prevention):
       log P(Class | Document) ∝ log P(Class) + ∑ log P(wi | Class)

5. Softmax / Posterior Odds:
       Converts unnormalized log-scores back to calibrated probabilities [0, 1].
"""

import math
from collections import Counter
from typing import Dict, List, Tuple, Any
from .text_processor import TextProcessor


class NaiveBayesClassifier:
    """Multinomial Naive Bayes text classifier with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0, text_processor: TextProcessor = None):
        self.alpha = alpha  # Laplace smoothing parameter
        self.processor = text_processor or TextProcessor()

        # Class counts
        self.doc_counts: Dict[str, int] = {"spam": 0, "ham": 0}
        self.total_docs: int = 0

        # Word counts per class
        self.word_counts: Dict[str, Counter] = {"spam": Counter(), "ham": Counter()}
        self.total_words: Dict[str, int] = {"spam": 0, "ham": 0}

        # Unique vocabulary across all classes
        self.vocab: set = set()

        # Prior probabilities
        self.log_priors: Dict[str, float] = {}

    def fit(self, texts: List[str], labels: List[str]) -> "NaiveBayesClassifier":
        """
        Trains the classifier on labeled texts.
        labels must be 'spam' or 'ham'.
        """
        self.doc_counts = {"spam": 0, "ham": 0}
        self.word_counts = {"spam": Counter(), "ham": Counter()}
        self.total_words = {"spam": 0, "ham": 0}
        self.vocab = set()

        # Phase 1: Aggregate counts
        for text, label in zip(texts, labels):
            label = label.lower().strip()
            if label not in ("spam", "ham"):
                continue

            self.doc_counts[label] += 1
            tokens = self.processor.tokenize(text)

            for token in tokens:
                self.word_counts[label][token] += 1
                self.vocab.add(token)

        self.total_docs = sum(self.doc_counts.values())
        self.total_words["spam"] = sum(self.word_counts["spam"].values())
        self.total_words["ham"] = sum(self.word_counts["ham"].values())

        # Phase 2: Compute log priors: log(P(Class)) = log(N_class / N_total)
        for label in ("spam", "ham"):
            if self.doc_counts[label] > 0:
                self.log_priors[label] = math.log(self.doc_counts[label] / self.total_docs)
            else:
                self.log_priors[label] = -float("inf")

        return self

    def word_likelihood(self, word: str, label: str) -> float:
        """
        Calculates P(word | label) with Laplace smoothing:
            P(w | c) = (count(w, c) + alpha) / (total_words_c + alpha * |V|)
        """
        vocab_size = len(self.vocab)
        count = self.word_counts[label].get(word, 0)
        denominator = self.total_words[label] + (self.alpha * vocab_size)
        return (count + self.alpha) / denominator

    def log_word_likelihood(self, word: str, label: str) -> float:
        """Returns log(P(word | label))."""
        return math.log(self.word_likelihood(word, label))

    def predict_log_scores(self, text: str) -> Tuple[float, float, List[str]]:
        """
        Computes unnormalized log-posteriors:
            log P(c | d) = log P(c) + sum_{w in d} log P(w | c)
        Only considers words present in the training vocabulary.
        """
        tokens = self.processor.tokenize(text)
        
        # Consider tokens present in vocabulary
        known_tokens = [t for t in tokens if t in self.vocab]

        log_spam = self.log_priors["spam"]
        log_ham = self.log_priors["ham"]

        for token in known_tokens:
            log_spam += self.log_word_likelihood(token, "spam")
            log_ham += self.log_word_likelihood(token, "ham")

        return log_spam, log_ham, known_tokens

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Calculates calibrated posterior probabilities using numerically stable softmax:
            P(spam | text) = 1 / (1 + exp(log_ham - log_spam))
        """
        log_spam, log_ham, _ = self.predict_log_scores(text)

        # Numerically stable conversion: subtract max
        max_log = max(log_spam, log_ham)
        exp_spam = math.exp(log_spam - max_log)
        exp_ham = math.exp(log_ham - max_log)
        total_exp = exp_spam + exp_ham

        p_spam = exp_spam / total_exp
        p_ham = exp_ham / total_exp

        return {"spam": p_spam, "ham": p_ham}

    def predict(self, text: str) -> str:
        """Predicts class: 'Spam' or 'Not Spam'."""
        proba = self.predict_proba(text)
        return "Spam" if proba["spam"] >= 0.5 else "Not Spam"

    def inspect_prediction(self, text: str) -> Dict[str, Any]:
        """
        Provides detailed mathematical breakdown showing how individual words
        influence the final Spam vs Not Spam decision.
        """
        log_spam, log_ham, tokens = self.predict_log_scores(text)
        proba = self.predict_proba(text)

        word_contributions = []
        for word in tokens:
            p_w_spam = self.word_likelihood(word, "spam")
            p_w_ham = self.word_likelihood(word, "ham")
            ratio = p_w_spam / p_w_ham  # Likelihood ratio (>1 means word leans spam)
            log_diff = math.log(p_w_spam) - math.log(p_w_ham)

            word_contributions.append({
                "word": word,
                "p_word_given_spam": p_w_spam,
                "p_word_given_ham": p_w_ham,
                "likelihood_ratio": ratio,
                "log_diff": log_diff,
                "direction": "Spam" if ratio > 1.0 else ("Not Spam" if ratio < 1.0 else "Neutral")
            })

        # Sort words by absolute impact on odds
        word_contributions.sort(key=lambda x: abs(x["log_diff"]), reverse=True)

        return {
            "prediction": "Spam" if proba["spam"] >= 0.5 else "Not Spam",
            "spam_probability": proba["spam"],
            "ham_probability": proba["ham"],
            "log_spam_score": log_spam,
            "log_ham_score": log_ham,
            "tokens_used": tokens,
            "word_breakdown": word_contributions,
            "prior_spam": math.exp(self.log_priors["spam"]),
            "prior_ham": math.exp(self.log_priors["ham"])
        }

    def top_indicator_words(self, top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Identifies words with the strongest Spam vs Ham likelihood ratios.
        Odds Ratio = P(w | Spam) / P(w | Ham)
        """
        ratios = []
        for word in self.vocab:
            p_spam = self.word_likelihood(word, "spam")
            p_ham = self.word_likelihood(word, "ham")
            ratios.append((word, p_spam / p_ham))

        # Top spam indicators have high ratio
        spam_indicators = sorted(ratios, key=lambda x: x[1], reverse=True)[:top_n]
        # Top ham indicators have low ratio
        ham_indicators = sorted(ratios, key=lambda x: x[1])[:top_n]

        # Convert ham ratio to inverse (Ham / Spam) for readable display
        ham_indicators = [(w, 1.0 / r) for w, r in ham_indicators]

        return {
            "top_spam_words": spam_indicators,
            "top_ham_words": ham_indicators
        }

