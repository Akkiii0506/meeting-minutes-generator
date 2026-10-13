import re
import json
from collections import defaultdict
from keybert import KeyBERT
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ---------- Decisions ----------
DECISION = re.compile(
    r"\b(?:we|i|team|everyone)(?:[’']ve| have)?\s+(?:decided|agreed|concluded|resolved)\b|"
    r"\b(?:decided|agreed|approved|finali[sz]ed|confirmed)\s+(?:to|on|that)\b|"
    r"\b(?:let[’']s|we[’']ll|we will|we[’']re going to|we are going to)\s+"
    r"(?:go with|switch|use|adopt|choose|stick with|postpone|delay|skip|drop|prioriti[sz]e|revert)\b|"
    r"\b(?:the|final)\s+(?:decision|plan)\s+is\b|"
    r"\b(?:it[’']s|that[’']s)\s+(?:settled|decided|final)\b",
    re.IGNORECASE,
)

def extract_decisions(sentences):
    decisions = []
    for i, s in enumerate(sentences):
        if len(s.split()) < 4 or s.rstrip().endswith("?"):
            continue
        if DECISION.search(s):
            decisions.append({"sentence_id": i, "decision": s})
    return decisions

# ---------- Keywords and topics ----------
EXTRA_STOP = {
    "agreed", "agree", "team", "meeting", "sprint", "okay", "thanks", "thank",
    "sure", "yes", "right", "let", "going", "just", "like", "need", "know",
    "think", "really", "good", "great", "today", "tomorrow", "yesterday", "devs",
    "absolutely", "fantastic", "excellent", "ensuring", "ensure", "looks",
    "resolving", "pending", "reported",
}
STOP_WORDS = list(ENGLISH_STOP_WORDS | EXTRA_STOP)

_kb = None

def get_kb():
    global _kb
    if _kb is None:
        _kb = KeyBERT("all-MiniLM-L6-v2")
    return _kb

def chunk_sentences(sentences, size=12):
    return [" ".join(sentences[i:i + size]) for i in range(0, len(sentences), size)]

def _overlaps(phrase, chosen):
    return any(phrase in c or c in phrase for c in chosen)

def extract_keywords(sentences, top_n=10, chunk_size=12, max_topics=6):
    kb = get_kb()
    scores = defaultdict(float)
    chunk_results = []

    for chunk in chunk_sentences(sentences, chunk_size):
        kws = kb.extract_keywords(
            chunk,
            keyphrase_ngram_range=(1, 2),
            stop_words=STOP_WORDS,
            use_mmr=True, diversity=0.5, top_n=5,
        )
        chunk_results.append(kws)
        for phrase, score in kws:
            scores[phrase.lower()] += score

    # Keywords: highest total score, skipping near-duplicates
    keywords = []
    for phrase, _ in sorted(scores.items(), key=lambda x: -x[1]):
        if not _overlaps(phrase, keywords):
            keywords.append(phrase)
        if len(keywords) == top_n:
            break

    # Topics: best distinct phrase per chunk, then keep only the strongest
    candidates, used = [], set()
    for idx, kws in enumerate(chunk_results):
        for phrase, score in kws:
            words = set(phrase.lower().split())
            if not words & used:
                candidates.append((idx, phrase.lower(), score))
                used |= words
                break

    best = sorted(candidates, key=lambda c: -c[2])[:max_topics]    # top by score
    topics = [p for _, p, _ in sorted(best, key=lambda c: c[0])]   # back in meeting order

    return keywords, topics

# ---------- Run from the command line ----------
if __name__ == "__main__":
    import sys
    path = sys.argv[1]   # a *_sentences.txt file
    with open(path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    decisions = extract_decisions(sentences)
    keywords, topics = extract_keywords(sentences)

    print(f"Decisions found: {len(decisions)}")
    for n, d in enumerate(decisions, 1):
        print(f"  {n}. {d['decision']}")

    print("\nKeywords:", ", ".join(keywords))
    print("\nDiscussion topics (in order):")
    for n, t in enumerate(topics, 1):
        print(f"  {n}. {t}")

    out_path = path.replace("_sentences.txt", "_insights.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"decisions": decisions, "keywords": keywords, "topics": topics},
                  f, indent=2, ensure_ascii=False)
    print("\nSaved to", out_path)