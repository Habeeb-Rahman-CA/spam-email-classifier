"""
demo.py - Comprehensive Walkthrough of Spam Classification, Probability & Text Processing
"""

import math
from src.text_processor import TextProcessor
from src.naive_bayes import NaiveBayesClassifier
from train_and_evaluate import load_dataset


def run_demo():
    print("=" * 75)
    print(" 📧 SPAM EMAIL CLASSIFIER: PROBABILITY & TEXT PROCESSING WALKTHROUGH")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. TEXT PROCESSING LESSON
    # -------------------------------------------------------------
    print("\n" + "─" * 75)
    print(" 🧠 PART 1: TEXT PROCESSING PIPELINE")
    print("─" * 75)
    print("Raw text cannot be fed directly into math equations. We must transform it:")
    sample_text = "URGENT!! Claim your $1,000 Walmart Gift Card at https://claim-rewards.com now!"
    processor = TextProcessor()
    prep = processor.explain_processing(sample_text)

    print(f"\n1. Original Raw Email:")
    print(f"   \"{prep['original_text']}\"")
    print(f"\n2. Normalization (Lowercasing, URLs -> _url_, Money -> _money_):")
    print(f"   \"{prep['normalized_text']}\"")
    print(f"\n3. Tokenization (Extracting individual words):")
    print(f"   {prep['all_tokens']}")
    print(f"\n4. Stop Word Removal (Removing uninformative words like 'your', 'at'):")
    print(f"   Filtered out: {prep['removed_stopwords']}")
    print(f"\n5. Final Feature Tokens used for modeling:")
    print(f"   {prep['final_tokens']}")

    # -------------------------------------------------------------
    # 2. PROBABILITY FOUNDATIONS
    # -------------------------------------------------------------
    print("\n" + "─" * 75)
    print(" 🧠 PART 2: PRACTICAL PROBABILITY & BAYES' THEOREM")
    print("─" * 75)
    print("""
    Bayes' Theorem:
                      P(Spam) * P(Email | Spam)
    P(Spam | Email) = ─────────────────────────
                              P(Email)

    1. Prior Probability P(Spam):
       The baseline probability that any random incoming email is spam before reading it.
       P(Spam) = (Number of Spam Emails) / (Total Emails)

    2. Likelihood P(Email | Spam):
       Under the 'Naive' conditional independence assumption:
       P(word1, word2, ..., wordN | Spam) = P(word1 | Spam) * P(word2 | Spam) * ...

    3. Laplace Smoothing (Why we need +1):
       If an email has the word 'bitcoin', and it was never in our ham training emails:
       P('bitcoin' | Ham) would be 0.
       Multiplying by 0 makes P(Ham | Email) = 0 regardless of all other words!
       Laplace smoothing adds α=1 to every word count:
           P(word | Class) = (count + 1) / (Total Words in Class + |Vocabulary|)

    4. Log-Probabilities (Numerical Underflow):
       Multiplying 50 probabilities like (0.001 * 0.0004 * ...) leads to 0.0 in computer memory.
       We take the log to convert multiplication into addition:
           log P(Class | Email) ∝ log P(Class) + ∑ log P(word | Class)
    """)

    # -------------------------------------------------------------
    # 3. TRAINING & LIVE CLASSIFICATION
    # -------------------------------------------------------------
    print("─" * 75)
    print(" 🧠 PART 3: TRAINING ON EMAIL DATASET & LIVE PREDICTIONS")
    print("─" * 75)

    texts, labels = load_dataset("data/emails.csv")
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(texts, labels)

    p_spam = math.exp(clf.log_priors["spam"])
    p_ham = math.exp(clf.log_priors["ham"])
    print(f"Trained on {len(texts)} emails ({clf.doc_counts['spam']} Spam, {clf.doc_counts['ham']} Not Spam).")
    print(f"Learned Prior P(Spam)     = {p_spam:.2f}")
    print(f"Learned Prior P(Not Spam) = {p_ham:.2f}")
    print(f"Vocabulary Size           = {len(clf.vocab)} unique tokens")

    # Compare 2 test emails
    test_cases = [
        "Urgent: You won $2,000 cash. Claim your prize immediately by clicking the link!",
        "Hi John, can we reschedule our sprint review meeting to tomorrow at 2 PM?"
    ]

    for email in test_cases:
        inspection = clf.inspect_prediction(email)
        pred = inspection["prediction"]
        spam_prob = inspection["spam_probability"] * 100
        ham_prob = inspection["ham_probability"] * 100

        print("\n" + "═" * 75)
        print(f"Testing Email: \"{email}\"")
        print(f"Prediction   : {pred.upper()} ({'🚨 Spam' if pred == 'Spam' else '🟢 Not Spam'})")
        print(f"Probabilities: P(Spam) = {spam_prob:.2f}% | P(Not Spam) = {ham_prob:.2f}%")
        print("\nSignificant tokens and their impact:")
        for w in inspection["word_breakdown"][:4]:
            multiplier = w["likelihood_ratio"] if w["direction"] == "Spam" else (1.0 / w["likelihood_ratio"])
            print(f"  • '{w['word']}': {multiplier:.2f}x more likely in {w['direction'].upper()}")

    print("\n" + "=" * 75)
    print(" 🚀 READY TO EXPLORE!")
    print(" • Run `python3 interactive_predict.py` to test your own emails")
    print(" • Run `python3 train_and_evaluate.py` to inspect training metrics & top indicator words")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_demo()

