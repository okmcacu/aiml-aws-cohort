# Break the Spam Filter

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/okmcacu/aiml-aws-cohort/blob/main/ML-WEEK3/spam-filter-lab/spam_filter.ipynb)

AWS AI/ML Week 3 hands-on task: train the 10-line spam detector from the slides, then try to fool it and explain why it fails.

## Dataset

[SMS Spam Collection (UCI)](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): 5,574 labelled text messages (4,827 ham, 747 spam).
The data is **not stored in this repo**. The notebook and the script download a tab-separated copy of it (from the
[`justmarkham/pycon-2016-tutorial`](https://github.com/justmarkham/pycon-2016-tutorial) repo) into `data/` the first time they run.

Some messages contain a `"` character, so the file is read with `quoting=csv.QUOTE_NONE`. With pandas' default settings neighbouring messages get merged and 2 rows are silently lost.

## Model

TF-IDF features + Multinomial Naive Bayes, 80/20 train/test split with `random_state=42` (the code from slide 6).

| Metric | Score |
|---|---|
| Train accuracy | 0.9733 |
| Test accuracy | 0.9596 |
| Precision (spam) | 1.0000 |
| Recall (spam) | 0.7205 |
| "Always say ham" baseline accuracy | 0.8556 |

Accuracy looks great, but recall shows the model misses 45 of the 161 spam messages in the test set (116 caught, and 0 good messages flagged).

**Limitation:** the dataset contains 403 duplicate rows, which I kept. Identical messages can appear in both the training and test sets, so the test scores may be slightly optimistic compared with genuinely new messages.

## Breaking it

| My message | Model said | Should be | Why it failed |
|---|---|---|---|
| `Cl@im your fr33 g1ft card now` | ham (p(spam) 0.075) | spam | Spelling trick. `fr33` and `g1ft` were never seen in training, and `Cl@im` is split into `cl` + `im`, where `im` is a strongly ham word. |
| `Y O U   W O N   F R E E   M O N E Y   C L I C K   H E R E` | ham (0.131) | spam | Spaced-out letters. The default tokenizer ignores one-letter tokens, so the message has zero features and the model falls back to its prior (about 87% of training messages are ham). |
| `Dear customer, your account has been selected for a reward. Kindly confirm your details at your earliest convenience.` | ham (0.202) | spam | Polite, businesslike tone. The spam-leaning words (`customer`, `reward`, `selected`) are outweighed by ordinary words that lean ham (`dear`, `at`, `for`, `confirm`). |
| `Reply STOP to cancel your bank statement alerts. Standard rates apply. Call customer care free of charge` | spam (0.820) | ham | A real service message reuses spam vocabulary (`apply`, `statement`, `customer`, `rates`, `standard`, `stop`), and `free` appears in 21.7% of spam training messages but only 1.3% of ham. The model counts words and cannot tell who is sending or why. |
| `Mobile top up of 100 birr received. Your new balance is 250 birr. Thank you for using our service` | spam (0.632) | ham | Numbers and words like `mobile`, `service` and `balance` appear mostly in spam in this dataset, and `birr` never appears in the training data at all. |

**Tried, but did not fool it:** `Not free, just confirming lunch at 1pm` is correctly classified as ham (p(spam) 0.026).
The model does not understand "not", but the ham-leaning words around it (`not`, `at`, `lunch`) outweigh `free`.

## What I learned

A word-counting model only knows the words it saw in training: spelling tricks, spaced-out letters and polite wording hide spam from it, and ordinary service messages that reuse spam words get flagged. High accuracy hid this because most messages are not spam, so I should also look at precision and recall and test with messages written differently from the training data.

## Run it

From the root of this repo:

```bash
cd ML-WEEK3/spam-filter-lab
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then either:

- run the notebook: open `spam_filter.ipynb` in VS Code (select the `venv` kernel), or click the Colab badge above and choose **Runtime > Run all**
- run the script: `python spam_filter.py`, or test your own messages with `python spam_filter.py "your message here"`

## Files

- `spam_filter.ipynb`: code, results and explanations
- `spam_filter.py`: the same model as a script, with an option to test your own messages
- `README.md`: this write-up
- `requirements.txt`: dependencies