"""
Unit tests for TextProcessor and NaiveBayesClassifier.
"""

import unittest
import math
from src.text_processor import TextProcessor
from src.naive_bayes import NaiveBayesClassifier


class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()

    def test_clean_text_normalizes_urls_and_money(self):
        raw = "Check https://prize-winner.com to get $1,000 now!"
        cleaned = self.processor.clean_text(raw)
        self.assertIn("_url_", cleaned)
        self.assertIn("_money_", cleaned)

    def test_tokenize_filters_stopwords(self):
        raw = "This is a meeting for the project"
        tokens = self.processor.tokenize(raw)
        self.assertIn("meeting", tokens)
        self.assertIn("project", tokens)
        self.assertNotIn("this", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("the", tokens)


class TestNaiveBayesClassifier(unittest.TestCase):
    def setUp(self):
        self.train_texts = [
            "Win free cash prize today only",
            "Urgent account alert claim money now",
            "Please review the meeting notes for tomorrow",
            "Attached is the project status report for team"
        ]
        self.train_labels = ["spam", "spam", "ham", "ham"]
        self.clf = NaiveBayesClassifier(alpha=1.0)
        self.clf.fit(self.train_texts, self.train_labels)

    def test_prior_probabilities(self):
        p_spam = math.exp(self.clf.log_priors["spam"])
        p_ham = math.exp(self.clf.log_priors["ham"])
        self.assertAlmostEqual(p_spam, 0.5)
        self.assertAlmostEqual(p_ham, 0.5)

    def test_spam_prediction(self):
        test_email = "Get free cash now"
        pred = self.clf.predict(test_email)
        proba = self.clf.predict_proba(test_email)
        self.assertEqual(pred, "Spam")
        self.assertGreater(proba["spam"], proba["ham"])

    def test_ham_prediction(self):
        test_email = "Can you send the meeting report tomorrow?"
        pred = self.clf.predict(test_email)
        proba = self.clf.predict_proba(test_email)
        self.assertEqual(pred, "Not Spam")
        self.assertGreater(proba["ham"], proba["spam"])

    def test_laplace_smoothing_prevents_zero(self):
        # A word never seen in training should still have a positive likelihood
        p_unseen_spam = self.clf.word_likelihood("cryptocurrency", "spam")
        p_unseen_ham = self.clf.word_likelihood("cryptocurrency", "ham")
        self.assertGreater(p_unseen_spam, 0.0)
        self.assertGreater(p_unseen_ham, 0.0)


if __name__ == "__main__":
    unittest.main()

