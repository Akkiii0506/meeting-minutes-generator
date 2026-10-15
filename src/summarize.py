import os
import re
import sys
import time
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODELS = {
    "flan-t5-base": "google/flan-t5-base",
    "flan-t5-large": "google/flan-t5-large",
    "distilbart": "sshleifer/distilbart-cnn-12-6",
    "bart-large": "facebook/bart-large-cnn",
    "bart-samsum": "philschmid/bart-large-cnn-samsum",
    "pegasus-xsum": "google/pegasus-xsum",
}
INPUT_LIMIT = {"flan-t5-base": 512, "flan-t5-large": 512, "distilbart": 1024,
               "bart-large": 1024, "bart-samsum": 1024, "pegasus-xsum": 512}

CHUNK_PROMPT = "Write a short third-person summary of this meeting excerpt:\n\n"
FINAL_PROMPT = "Write a short third-person summary of this meeting based on these notes:\n\n"

_cache = {}

def load_model(name):
    """Load a model once and reuse it."""
    if name not in _cache:
        path = MODELS[name]
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        _cache[name] = (tok, model)
    return _cache[name]

def chunk_by_tokens(sentences, tok, max_tokens=450):
    """Group whole sentences into chunks of at most max_tokens tokens."""
    chunks, current, length = [], [], 0
    for s in sentences:
        n = len(tok.encode(s, add_special_tokens=False))
        if current and length + n > max_tokens:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(s)
        length += n
    if current:
        chunks.append(" ".join(current))
    return chunks

def trim_incomplete(text):
    """Cut off a half-finished last sentence."""
    ends = list(re.finditer(r"[.!?]", text))
    if text and text[-1] not in ".!?" and ends:
        return text[:ends[-1].end()]
    return text

def fix_spacing(text):
    """Remove the space the tokenizer leaves before punctuation."""
    return re.sub(r"\s+([.,!?;:])", r"\1", text)

def neutralize_first_person(text):
    """Replace 'I', 'my', 'me' with 'the speaker' so the summarizer does not
    guess which named person 'I' refers to."""
    rules = [
        (r"\bI[’']ll\b", "the speaker will"),
        (r"\bI[’']ve\b", "the speaker has"),
        (r"\bI[’']m\b", "the speaker is"),
        (r"\bI[’']d\b", "the speaker would"),
        (r"\bI am\b", "the speaker is"),
        (r"\bI have\b", "the speaker has"),
        (r"\bI was\b", "the speaker was"),
        (r"\bI\b", "the speaker"),
        (r"\bmyself\b", "the speaker"),
        (r"\bMy\b", "The speaker's"),
        (r"\bmy\b", "the speaker's"),
        (r"\bme\b", "the speaker"),
    ]
    for pattern, repl in rules:
        text = re.sub(pattern, repl, text)
    return text[:1].upper() + text[1:]

def summarize_text(text, name, prompt="", max_new=60, min_new=20):
    tok, model = load_model(name)
    if "t5" in name:                       # only T5 models use instructions
        text = prompt + text
    inputs = tok(text, return_tensors="pt",
                 truncation=True, max_length=INPUT_LIMIT[name])
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            num_beams=4,
            no_repeat_ngram_size=3,
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            early_stopping=True,
        )
    return trim_incomplete(fix_spacing(tok.decode(out[0], skip_special_tokens=True).strip()))

def summarize_meeting(sentences, name="flan-t5-base", keywords=None,
                      chunk_tokens=450, second_pass=False, neutral_speaker=False):
    # Questions and very short sentences carry little information
    sentences = [s for s in sentences if not s.rstrip().endswith("?") and len(s.split()) >= 4]
    if neutral_speaker:
        sentences = [neutralize_first_person(s) for s in sentences]

    tok, _ = load_model(name)
    chunks = chunk_by_tokens(sentences, tok, chunk_tokens)
    partials = [summarize_text(c, name, CHUNK_PROMPT) for c in chunks]

    if second_pass and len(partials) > 1:
        notes = " ".join(partials)
        if keywords:
            notes = f"Main topics: {', '.join(keywords[:6])}.\n\n" + notes
        final = summarize_text(notes, name, FINAL_PROMPT, max_new=150, min_new=50)
    else:
        final = " ".join(partials)        # chunk summaries cover the whole meeting
    return {"model": name, "chunks": len(chunks), "partials": partials, "final": final}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    neutral = "--neutral" in sys.argv
    path = args[0]                                       # a *_sentences.txt file
    name = args[1] if len(args) > 1 else "flan-t5-base"

    with open(path, encoding="utf-8-sig") as f:
        sentences = [line.strip() for line in f if line.strip()]

    keywords = None
    insights_path = path.replace("_sentences.txt", "_insights.json")
    if os.path.exists(insights_path):
        with open(insights_path, encoding="utf-8-sig") as f:
            keywords = json.load(f).get("keywords")

    start = time.time()
    result = summarize_meeting(sentences, name, keywords, neutral_speaker=neutral)
    elapsed = time.time() - start

    label = f"{name} (speaker-neutral)" if neutral else name
    print(f"Model: {label} | chunks: {result['chunks']} | time: {elapsed:.1f}s\n")
    for i, p in enumerate(result["partials"], 1):
        print(f"Chunk {i}: {p}\n")
    print("FINAL SUMMARY:\n", result["final"])

    base = os.path.basename(path).replace("_sentences.txt", "")
    suffix = f"{name}-neutral" if neutral else name
    out_path = os.path.join(os.path.dirname(path), f"{base}_summary_{suffix}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result["final"])
    print("\nSaved to", out_path)