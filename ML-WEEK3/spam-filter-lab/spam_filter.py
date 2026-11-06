"""Break the Spam Filter - AWS AI/ML Week 3 hands-on task.

Trains the 10-line spam detector from the slides (TF-IDF + Multinomial Naive Bayes) on the
SMS Spam Collection, prints its scores, then tests messages that try to fool it.

Usage:
    python spam_filter.py                       # train, score, run the 5 built-in tricks
    python spam_filter.py "your message" ...    # train, score, test your own messages
"""
import csv
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sms.tsv")

# (message, what it really is). Three spam messages that try to slip through,
# two normal messages that try to get flagged.
TRICKS = [
    ("Cl@im your fr33 g1ft card now", "spam"),
    ("Y O U   W O N   F R E E   M O N E Y   C L I C K   H E R E", "spam"),
    ("Dear customer, your account has been selected for a reward. Kindly confirm your details at your earliest convenience.", "spam"),
    ("Reply STOP to cancel your bank statement alerts. Standard rates apply. Call customer care free of charge", "ham"),
    ("Mobile top up of 100 birr received. Your new balance is 250 birr. Thank you for using our service", "ham"),
]


def load_data(path=DATA_PATH):
    """Download the dataset on first run, then load it (label, message)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, path)
    # QUOTE_NONE: some messages contain a " character; with pandas' default quote handling
    # neighbouring messages get merged and rows are silently dropped (5,572 instead of 5,574).
    return pd.read_csv(path, sep="\t", header=None, names=["label", "message"],
                       quoting=csv.QUOTE_NONE)


def train(df):
    """The 10-line model from slide 6. Returns the vectorizer, the model and the split."""
    emails, labels = df["message"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.2, random_state=42)

    vec = TfidfVectorizer()          # text -> numbers (features)
    model = MultinomialNB()          # a simple, fast classifier
    model.fit(vec.fit_transform(X_train), y_train)   # learn
    return vec, model, (X_train, X_test, y_train, y_test)


def evaluate(vec, model, split):
    """Slide 10: compare train and test scores, plus precision and recall for spam."""
    X_train, X_test, y_train, y_test = split
    pred_tr = model.predict(vec.transform(X_train))
    pred_te = model.predict(vec.transform(X_test))
    print("train acc:", round(accuracy_score(y_train, pred_tr), 4))
    print("test acc: ", round(accuracy_score(y_test, pred_te), 4))
    print("precision:", round(precision_score(y_test, pred_te, pos_label="spam"), 4))
    print("recall:   ", round(recall_score(y_test, pred_te, pos_label="spam"), 4))
    print("always-'ham' baseline accuracy:", round((y_test == "ham").mean(), 4))


def predict(vec, model, messages):
    """Return the predicted labels and the probability of spam for each message."""
    features = vec.transform(messages)
    spam_col = list(model.classes_).index("spam")
    return model.predict(features), model.predict_proba(features)[:, spam_col]


def analyse(vec, model, message, top=6):
    """Work out what the model sees: unknown words, and which words lean spam or ham."""
    classes = list(model.classes_)
    spam_i, ham_i = classes.index("spam"), classes.index("ham")
    tokens = vec.build_analyzer()(message)
    unknown = sorted({t for t in tokens if t not in vec.vocabulary_})
    # log P(word | spam) - log P(word | ham): positive leans spam, negative leans ham
    lean = {
        t: float(model.feature_log_prob_[spam_i, vec.vocabulary_[t]]
                 - model.feature_log_prob_[ham_i, vec.vocabulary_[t]])
        for t in set(tokens) if t in vec.vocabulary_
    }
    ranked = sorted(lean.items(), key=lambda kv: kv[1], reverse=True)
    spammy = [(w, round(v, 2)) for w, v in ranked if v > 0][:top]
    hammy = [(w, round(v, 2)) for w, v in ranked[::-1] if v < 0][:top]
    return tokens, unknown, lean, spammy, hammy


def explain(vec, model, message, top=6):
    """Show what the model sees, for one message."""
    tokens, unknown, _, spammy, hammy = analyse(vec, model, message, top)
    print("    words the model sees  :", tokens if tokens else "NONE")
    print("    non-zero features     :", vec.transform([message]).nnz)
    print("    never seen in training:", unknown if unknown else "-")
    print("    leans spam            :", spammy if spammy else "-")
    print("    leans ham             :", hammy if hammy else "-")


def summary(vec, model, tests, labels, p_spam):
    """Plain-language recap printed at the very bottom."""
    classes = list(model.classes_)
    ham_prior = float(np.exp(model.class_log_prior_[classes.index("ham")]))
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    fooled = 0
    for n, ((message, should_be), label, p) in enumerate(zip(tests, labels, p_spam), 1):
        _, unknown, lean, spammy, hammy = analyse(vec, model, message, top=3)
        confidence = max(p, 1 - p)
        line = f"{n}. {label.upper()}  (p(spam) = {p:.3f}, {confidence:.0%} sure)"
        if should_be is not None:
            if label != should_be:
                fooled += 1
                line += f"  -> FOOLED, it should be {should_be}"
            else:
                line += "  -> correct"
        print(line)
        print("   message  :", " ".join(message.split())[:60])
        total = len(lean) + len(unknown)
        if not lean:
            print("   reading  : NO known words, so it fell back to the prior "
                  f"({ham_prior:.0%} ham)")
        else:
            print(f"   reading  : {len(lean)} known words used, "
                  f"{len(unknown)} of {total} distinct words ignored (never seen in training)")
            print("   spam cues:", ", ".join(w for w, _ in spammy) or "-")
            print("   ham cues :", ", ".join(w for w, _ in hammy) or "-")
    if any(should_be is not None for _, should_be in tests):
        print(f"\nFooled the model: {fooled} of {len(tests)} messages.")
    print(f"Starting point before reading any words: {ham_prior:.0%} ham, "
          f"{1 - ham_prior:.0%} spam.")
    print("A word counter: it never understands the message, it only adds up word clues.")


def main(argv):
    df = load_data()
    print("Messages:", df.shape[0], "|", df["label"].value_counts().to_dict())

    vec, model, split = train(df)
    print()
    evaluate(vec, model, split)

    classes = list(model.classes_)
    print("\nPrior (before reading any words): ham =",
          round(float(np.exp(model.class_log_prior_[classes.index("ham")])), 3),
          "| spam =", round(float(np.exp(model.class_log_prior_[classes.index("spam")])), 3))

    if argv:
        tests = [(m, None) for m in argv]
        print("\n=== Your messages ===")
    else:
        tests = TRICKS
        print("\n=== Break it: five messages ===")

    messages = [m for m, _ in tests]
    labels, p_spam = predict(vec, model, messages)
    for (message, should_be), label, p in zip(tests, labels, p_spam):
        if should_be is None:
            status, detail = "", f"model said {label} (p(spam) = {p:.3f})"
        else:
            status = "FOOLED  " if label != should_be else "correct "
            detail = f"should be {should_be} | model said {label} (p(spam) = {p:.3f})"
        print(f"\n{status}{detail}")
        print("  ", message[:90])
        explain(vec, model, message)

    summary(vec, model, tests, labels, p_spam)


if __name__ == "__main__":
    main(sys.argv[1:])