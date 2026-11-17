# TinyGPT

A small GPT-style language model built from scratch: trained on three classic
books and served as a chat page that runs entirely in the browser.

It is a learning/demo project. The model has under a million parameters, so it
writes text *in the style of* its books — it does not answer questions the way
ChatGPT does.

## How it works

| Step | File | What it does |
|------|------|--------------|
| 1 | `src/prepare.py` | Cleans the books in `data/raw/` and joins them into `data/input.txt` |
| 2 | `src/model.py` | The transformer: embeddings, self-attention, feed-forward blocks |
| 3 | `src/train.py` | Character-level training loop, saves `out/ckpt.pt` |
| 4 | `src/sample.py` | Generates text from the checkpoint in the terminal |
| 5 | `src/export.py` | Writes the weights to `docs/model/` for the browser |
| 6 | `docs/` | Chat page + the same model re-implemented in plain JavaScript |

Model size: 4 layers, 4 heads, 128-dim embeddings, 128-character context
(about 0.83M parameters).

## Books

All three are in the public domain (text from Project Gutenberg):

- *Alice's Adventures in Wonderland* — Lewis Carroll
- *Pride and Prejudice* — Jane Austen
- *The Adventures of Sherlock Holmes* — Arthur Conan Doyle

## Run it locally

```bash
pip install -r requirements.txt

python src/prepare.py          # build data/input.txt
python src/train.py            # ~15 min on a laptop CPU, faster on a GPU
python src/sample.py --prompt "Alice was"
python src/export.py           # write docs/model/
node tests/parity.mjs          # check JavaScript matches PyTorch

python -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>. The repository already includes trained
weights in `docs/model/`, so the last command works on its own.

## Deploy on GitHub Pages

Repository **Settings → Pages → Deploy from a branch →** branch `main`,
folder `/docs`. The page is fully static, so nothing else is needed.

## Train on your own books

Put any plain-text files in `data/raw/` and run the steps above again.
