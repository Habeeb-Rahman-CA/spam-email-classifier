# Spam Email Classifier: Learn Practical Probability & Text Processing

A from-scratch, dependency-free implementation of a **Multinomial Naive Bayes Spam Email Classifier** in Python.

Output: **Spam** / **Not Spam** (with calibrated probability confidence).

---

## 📚 1. Text Processing Pipeline

Raw email text cannot be directly computed by mathematical models. It passes through five key stages implemented in [`src/text_processor.py`](src/text_processor.py):

```
Raw Email Text
      │
      ▼
1. Case Normalization       ("WIN $1000 NOW!" -> "win $1000 now!")
      │
      ▼
2. Entity Normalization     (URLs -> _url_, $1000 -> _money_, 123 -> _number_)
      │
      ▼
3. Tokenization             (Extract word tokens using regex boundaries)
      │
      ▼
4. Stop Word Removal        (Filter frequent non-informative words: "the", "at", "is")
      │
      ▼
5. Vocabulary Representation (Feature tokens fed into the Naive Bayes model)
```

### Why normalize entities like `$1,000` to `_money_`?
If an email contains `$500`, `$1,000`, or `$250,000`, treating them as completely separate words splits their statistical weight across dozens of rare tokens. By mapping them all to `_money_`, the classifier learns that the **presence of currency figures** strongly correlates with spam.

---

## 🧮 2. Practical Probability & Naive Bayes

Implemented in [`src/naive_bayes.py`](src/naive_bayes.py).

### 1. Bayes' Theorem
We want to find whether an email is more likely **Spam** or **Not Spam (Ham)** given the words it contains:

$$P(\text{Spam} \mid \text{Email}) = \frac{P(\text{Spam}) \times P(\text{Email} \mid \text{Spam})}{P(\text{Email})}$$

### 2. Prior Probability $P(\text{Class})$
The baseline probability of an email being spam before we even inspect its text:

$$P(\text{Spam}) = \frac{\text{Count}(\text{Spam})}{\text{Total Emails}}, \quad P(\text{Ham}) = \frac{\text{Count}(\text{Ham})}{\text{Total Emails}}$$

### 3. The "Naive" Conditional Independence Assumption
Assuming that each word appears independently given the class:

$$P(w_1, w_2, \dots, w_n \mid \text{Class}) = \prod_{i=1}^{n} P(w_i \mid \text{Class})$$

### 4. Laplace (Additive) Smoothing ($\alpha = 1$)
**The Zero-Probability Problem**: If a word (e.g. `"bitcoin"`) appears in a test email but was never seen in the "Not Spam" training emails, $P(\text{bitcoin} \mid \text{Ham}) = 0$. Multiplying by 0 makes the entire email's Ham probability 0, ignoring all other legitimate words!

Laplace smoothing adds a pseudo-count $\alpha = 1$:

$$P(w \mid \text{Class}) = \frac{\text{Count}(w, \text{Class}) + 1}{\sum \text{Words in Class} + |V|}$$

*(where $|V|$ is the total number of unique words across the entire training vocabulary)*.

### 5. Log-Probabilities (Preventing Floating-Point Underflow)
Multiplying dozens of fractional probabilities ($0.001 \times 0.0004 \times \dots$) quickly rounds to `0.0` in 64-bit IEEE floating-point arithmetic. 

By applying $\log(\cdot)$, multiplication turns into stable addition:

$$\log P(\text{Class} \mid \text{Email}) \propto \log P(\text{Class}) + \sum_{i=1}^{n} \log P(w_i \mid \text{Class})$$

### 6. Calibrated Softmax Posterior Probabilities
To convert unnormalized log-scores back to an intuitive percentage (e.g. 99.4% Spam vs 0.6% Not Spam):

$$P(\text{Spam} \mid \text{Email}) = \frac{e^{\log P(\text{Spam} \mid \text{Email})}}{e^{\log P(\text{Spam} \mid \text{Email})} + e^{\log P(\text{Ham} \mid \text{Email})}} = \frac{1}{1 + e^{\log P(\text{Ham} \mid \text{Email}) - \log P(\text{Spam} \mid \text{Email})}}$$

---

## 🚀 Quickstart & Usage

### 1. Run the Interactive Demo
A guided visual walkthrough of the text pipeline, probability formulas, and live classification:
```bash
python3 demo.py
```

### 2. Train and Evaluate
Splits [`data/emails.csv`](data/emails.csv) into train/test sets, calculates Accuracy, Precision, Recall, F1-Score, and shows the top learned Spam vs. Ham indicator keywords:
```bash
python3 train_and_evaluate.py
```

### 3. Test Any Custom Email
Pass text directly as an argument:
```bash
python3 interactive_predict.py "Urgent: Reset your account password by clicking here immediately"
```
Or run interactively to test custom emails or preset scenarios:
```bash
python3 interactive_predict.py
```

### 4. Run Unit Tests
```bash
python3 -m unittest discover tests
```

---

## 📁 Project Structure

```
spam-email-classifier/
├── data/
│   └── emails.csv                 # Labeled training dataset (Spam / Ham)
├── src/
│   ├── __init__.py
│   ├── text_processor.py          # Tokenization, cleaning, entity normalization & stop words
│   └── naive_bayes.py             # From-scratch Naive Bayes with Laplace smoothing & log posteriors
├── tests/
│   └── test_spam_classifier.py    # Unit tests for text processing and probability calculation
├── train_and_evaluate.py          # Stratified split, metrics (Accuracy, Precision, Recall, F1), odds ratios
├── interactive_predict.py         # CLI tool to test custom emails and see probability breakdown
├── demo.py                        # Complete educational walkthrough
└── README.md                      # Documentation & theoretical breakdown
```

