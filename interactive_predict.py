"""
interactive_predict.py - Interactive Spam Email Classifier & Probability Inspector

Run this script to test any email text and see:
1. Prediction: 'Spam' or 'Not Spam'
2. Calibrated probabilities (e.g. 98.2% Spam)
3. Text processing steps
4. Word-by-word mathematical probability impact
"""

import sys
from src.text_processor import TextProcessor
from src.naive_bayes import NaiveBayesClassifier
from train_and_evaluate import load_dataset


def train_production_model():
    """Trains on the complete dataset for best interactive coverage."""
    texts, labels = load_dataset("data/emails.csv")
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(texts, labels)
    return clf


def analyze_email(clf: NaiveBayesClassifier, email_text: str):
    print("\n" + "=" * 70)
    print(" 📧 INPUT EMAIL:")
    print(f' "{email_text}"')
    print("=" * 70)

    # 1. Text processing inspection
    prep_info = clf.processor.explain_processing(email_text)
    print("\n[Step 1: Text Processing Pipeline]")
    print(f"  • Normalized text : {prep_info['normalized_text']}")
    print(f"  • Extracted tokens: {prep_info['all_tokens']}")
    print(f"  • Filtered stop words: {prep_info['removed_stopwords']}")
    print(f"  • Final tokens for model: {prep_info['final_tokens']}")

    # 2. Prediction and Probability
    inspection = clf.inspect_prediction(email_text)
    pred = inspection["prediction"]
    spam_pct = inspection["spam_probability"] * 100
    ham_pct = inspection["ham_probability"] * 100

    icon = "🚨 [SPAM]" if pred == "Spam" else "🟢 [NOT SPAM]"
    print(f"\n[Step 2: Classification Result]")
    print(f"  ==> Output: {pred.upper()} {icon}")
    print(f"  • Confidence: {max(spam_pct, ham_pct):.2f}%")
    print(f"  • Probability distribution: P(Spam) = {spam_pct:.2f}% | P(Not Spam) = {ham_pct:.2f}%")

    # 3. Mathematical Probability breakdown
    print(f"\n[Step 3: Probability Breakdown & Evidence]")
    print(f"  • Prior P(Spam) = {inspection['prior_spam']:.2f} | Prior P(Not Spam) = {inspection['prior_ham']:.2f}")
    print(f"  • Total Log-Likelihood Score: Spam={inspection['log_spam_score']:.2f} vs Ham={inspection['log_ham_score']:.2f}")
    
    if inspection["word_breakdown"]:
        print("\n  Word-by-word evidence (sorted by impact):")
        print(f"  {'Word':<15} {'P(w|Spam)':<12} {'P(w|Ham)':<12} {'Impact':<12} {'Leans'}")
        print("  " + "-" * 58)
        for w in inspection["word_breakdown"]:
            ratio_str = f"{w['likelihood_ratio']:.2f}x" if w["direction"] == "Spam" else f"{1.0/w['likelihood_ratio']:.2f}x"
            print(f"  {w['word']:<15} {w['p_word_given_spam']:<12.5f} {w['p_word_given_ham']:<12.5f} {ratio_str:<12} {w['direction']}")
    else:
        print("  (No known training vocabulary words found in text; decision is purely based on prior probability.)")
    print("=" * 70)


def interactive_cli():
    print("\nLoading dataset and training Naive Bayes model...")
    clf = train_production_model()
    print("Model ready!")

    sample_prompts = [
        "URGENT! You won $10,000 cash. Click here to claim your free reward immediately!",
        "Can we reschedule our project sync meeting to tomorrow morning? Let me know.",
        "Your PayPal account was locked due to suspicious login. Verify your password now at http://fake-login.com",
        "Please review the attached spreadsheet before the client presentation this afternoon."
    ]

    print("\nType your own email text, enter a sample number [1-4], or type 'exit' to quit:")
    for i, s in enumerate(sample_prompts, 1):
        print(f"  [{i}] \"{s}\"")

    while True:
        try:
            print("\nEnter email text (or 1-4, or 'exit'):")
            user_input = input(">> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting.")
                break

            if user_input in ("1", "2", "3", "4"):
                user_input = sample_prompts[int(user_input) - 1]

            analyze_email(clf, user_input)

        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Direct CLI argument invocation
        clf = train_production_model()
        analyze_email(clf, " ".join(sys.argv[1:]))
    else:
        interactive_cli()

