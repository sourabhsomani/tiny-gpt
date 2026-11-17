"""Clean the raw Project Gutenberg books and join them into one training file."""
import glob
import os
import re

RAW_DIR = os.path.join("data", "raw")
OUT_FILE = os.path.join("data", "input.txt")

START = re.compile(r"\*\*\* ?START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*")
END = re.compile(r"\*\*\* ?END OF (THIS|THE) PROJECT GUTENBERG EBOOK")


def clean(text):
    """Drop the Gutenberg licence header/footer and tidy whitespace."""
    text = text.replace("\r\n", "\n")
    start = START.search(text)
    if start:
        text = text[start.end():]
    end = END.search(text)
    if end:
        text = text[:end.start()]
    # keep plain ASCII only so the vocabulary stays small
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def main():
    books = sorted(glob.glob(os.path.join(RAW_DIR, "*.txt")))
    if not books:
        raise SystemExit(f"no books found in {RAW_DIR}")
    parts = []
    for path in books:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = clean(f.read())
        print(f"{os.path.basename(path):32s} {len(text):>9,} chars")
        parts.append(text)
    data = "\n\n".join(parts)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(data)
    print(f"wrote {OUT_FILE}: {len(data):,} chars, {len(set(data))} unique")


if __name__ == "__main__":
    main()
