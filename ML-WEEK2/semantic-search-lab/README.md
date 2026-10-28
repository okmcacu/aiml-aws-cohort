# Medical FAQ Semantic Search

A mini semantic search built for the AASTU AI/ML cohort, Week 2 lab.
It loads a CSV of medical FAQs, cleans it with Pandas, splits the answers into
overlapping chunks, embeds them with Chroma, and returns the closest chunks for a question.
It uses NumPy to measure answer lengths and to flag weak matches.

> Learning project. Not medical advice.

## Files

| File | What it is |
|---|---|
| `semantic_search_lab.ipynb` | The notebook (Colab or Jupyter), with explanations and tests |
| `semantic_search.py` | The same pipeline as a plain script |
| `data/faqs.csv` | 50 raw FAQ rows (1 missing answer, 1 duplicate, some extra spaces) |
| `requirements.txt` | Python packages |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Script (the first run downloads the embedding model, about 80 MB, so it can take a while):

```bash
python semantic_search.py
```

Notebook: open `semantic_search_lab.ipynb` in VS Code (select the `.venv` kernel) or in Colab,
then use Run all. In Colab, upload `faqs.csv` when asked.

## How it works

1. **Load** the CSV with Pandas.
2. **Clean**: strip extra spaces, drop rows with a missing question or answer, drop duplicate questions (50 rows become 48).
3. **Chunk**: split answers into 40-word pieces that overlap by 10 words, and keep the question at the start of each chunk (48 FAQs become 79 chunks).
4. **Embed and store**: Chroma turns each chunk into a vector with `all-MiniLM-L6-v2` and stores it.
5. **Search**: the query becomes a vector too, and Chroma returns the 3 closest chunks. Lower distance means closer.

## Example queries and results

Distance: lower is closer. Chunks of the same FAQ can appear more than once.

**"how do I avoid mosquito bites"**
1. (0.725) How can I prevent malaria? (chunk 1)
2. (1.103) How can I prevent malaria? (chunk 2)
3. (1.156) What should I do after a snake bite?

The right FAQ ranks first even though the query never says "malaria".

**"someone collapsed and is not breathing"**
1. (0.960) What should I do if someone faints?
2. (0.968) What should I do if someone faints? (chunk 2)
3. (0.990) How do I help a choking adult?

The CPR FAQ ("How do I perform CPR on an adult?") is in the data but did not appear in the top 3.
The distances are close together and none is low, so this is a weak result. See the limitations below.

**"I twisted my foot playing football and it is swollen"**
(paste your top 3 here)

**"I keep vomiting and have stomach cramps after eating"**
(paste your top 3 here)

## Limitations

- Chroma always returns the top 3, even when the last ones are not relevant. The notebook flags distances of 1.0 or more as weak matches (a practice threshold, not a scientific value).
- The small embedding model can miss a relevant FAQ when the wording differs a lot, as with the CPR example above.
- Only 48 FAQs, so many questions have no good answer in the data.