import re
import spacy

_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

# Filler words and phrases (longest first so phrases match before single words)
FILLERS = [
    r"you know", r"i mean", r"kind of", r"sort of",
    r"um+", r"uh+", r"erm", r"hmm+", r"ah+", r"like,",
]
FILLER_PATTERN = re.compile(r"\b(?:" + "|".join(FILLERS) + r")\b[,]?\s*", re.IGNORECASE)

def remove_fillers(text):
    return FILLER_PATTERN.sub("", text)

def remove_repeats(text):
    """'we we need' -> 'we need' (stutters from speech)."""
    return re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

def fix_spacing(text):
    text = re.sub(r"\s+", " ", text)                  # collapse spaces
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)      # no space before punctuation
    text = re.sub(r"([,.!?;:])(?=[A-Za-z])", r"\1 ", text)  # space after punctuation
    return text.strip()

def capitalize_sentences(text):
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)

def clean_text(text):
    """Cleaned text: keeps case and punctuation (for NER and summarization)."""
    text = remove_fillers(text)
    text = remove_repeats(text)
    text = fix_spacing(text)
    text = capitalize_sentences(text)
    return text

def split_sentences(text):
    doc = get_nlp()(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 1]

def normalize_tokens(text):
    """Lowercased, lemmatized content words (for keywords and statistics)."""
    doc = get_nlp()(text)
    return [t.lemma_.lower() for t in doc
            if t.is_alpha and not t.is_stop]

def preprocess(raw_text):
    cleaned = clean_text(raw_text)
    sentences = split_sentences(cleaned)
    tokens = normalize_tokens(cleaned)
    return {"cleaned": cleaned, "sentences": sentences, "tokens": tokens}

if __name__ == "__main__":
    import sys, os
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    result = preprocess(raw)

    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = "data/transcripts"

    with open(os.path.join(out_dir, f"{name}_clean.txt"), "w", encoding="utf-8") as f:
        f.write(result["cleaned"])
    with open(os.path.join(out_dir, f"{name}_sentences.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(result["sentences"]))

    print(f"Raw words:      {len(raw.split())}")
    print(f"Cleaned words:  {len(result['cleaned'].split())}")
    print(f"Sentences:      {len(result['sentences'])}")
    print(f"Content tokens: {len(result['tokens'])}")
    print("\nFirst 5 sentences:")
    for s in result["sentences"][:5]:
        print(" -", s)