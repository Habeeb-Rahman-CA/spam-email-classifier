"""
text_processor.py - Text Preprocessing Pipeline for Spam Classification

Core Steps:
1. Case Normalization (lowercasing)
2. Entity Normalization (URLs, emails, currency, numbers)
3. Tokenization (splitting raw text into individual words/tokens)
4. Stop Word Removal (filtering high-frequency non-informative terms)
5. Minimum Length Filtering (removing single characters / noise)
"""

import re
from typing import List, Set

# Standard English stop words
DEFAULT_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class TextProcessor:
    """Preprocesses raw email text into clean, normalized word tokens."""

    def __init__(
        self,
        normalize_entities: bool = True,
        remove_stopwords: bool = True,
        min_word_length: int = 2,
        custom_stopwords: Set[str] = None
    ):
        self.normalize_entities = normalize_entities
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.stopwords = custom_stopwords if custom_stopwords is not None else DEFAULT_STOP_WORDS

    def clean_text(self, text: str) -> str:
        """
        Applies regex replacements to normalize text patterns.
        Replaces specific patterns with informative placeholder tokens:
        - URLs -> _url_
        - Email addresses -> _email_
        - Currency / prices -> _money_
        - Numbers -> _number_
        """
        # Lowercase first
        text = text.lower()

        if self.normalize_entities:
            # Replace URLs (http://, https://, www., or generic domains)
            text = re.sub(r'https?://\S+|www\.\S+', ' _url_ ', text)

            # Replace Email addresses
            text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' _email_ ', text)

            # Replace Currency ($100, $5,000, 100$, 50€, etc.)
            text = re.sub(r'[\$€£¥]\s*\d+(?:,\d+)*(?:\.\d+)?|\b\d+(?:,\d+)*(?:\.\d+)?\s*[\$€£¥]', ' _money_ ', text)

            # Replace Standalone numbers
            text = re.sub(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', ' _number_ ', text)

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Extracts tokens from text using word boundary patterns.
        Retains entity tags like _url_, _money_, _number_, _email_.
        """
        cleaned = self.clean_text(text)
        
        # Match standard words or our special normalized tokens (_tag_)
        tokens = re.findall(r'_?[a-z]+_?', cleaned)

        # Filter stopwords and min length
        processed_tokens: List[str] = []
        for token in tokens:
            # Keep special entity tokens
            if token.startswith('_') and token.endswith('_'):
                processed_tokens.append(token)
                continue

            if len(token) < self.min_word_length:
                continue

            if self.remove_stopwords and token in self.stopwords:
                continue

            processed_tokens.append(token)

        return processed_tokens

    def explain_processing(self, raw_text: str) -> dict:
        """Returns step-by-step transformation for educational inspection."""
        step1_cleaned = self.clean_text(raw_text)
        raw_tokens = re.findall(r'_?[a-z]+_?', step1_cleaned)
        step2_filtered = [
            t for t in raw_tokens
            if (t.startswith('_') and t.endswith('_')) or (len(t) >= self.min_word_length)
        ]
        step3_no_stopwords = [
            t for t in step2_filtered
            if (t.startswith('_') and t.endswith('_')) or (t not in self.stopwords)
        ]
        removed_words = [t for t in step2_filtered if t in self.stopwords]

        return {
            "original_text": raw_text,
            "normalized_text": step1_cleaned,
            "all_tokens": raw_tokens,
            "removed_stopwords": removed_words,
            "final_tokens": step3_no_stopwords
        }

