import re
import spacy
from collections import Counter

WANTED_LABELS = {"PERSON", "ORG", "DATE", "TIME", "GPE", "DEADLINE"}
NOT_NAMES = {"dev", "devs", "team", "guys", "everyone", "folks", "qa", "ui", "ux"}
TITLES = r"^(mr|mrs|ms|miss|dr|prof)\.?\s+"

_ner_nlp = None

def get_ner_nlp():
    """spaCy pipeline with custom rules placed BEFORE the statistical NER."""
    global _ner_nlp
    if _ner_nlp is None:
        nlp = spacy.load("en_core_web_sm")
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns([
            # Tech terms: labelled TERM so they are not mistaken for ORG
            {"label": "TERM", "pattern": [{"LOWER": {"IN": [
                "ui", "ux", "api", "apis", "qa", "ci", "cd", "backend",
                "frontend", "sdk", "jira", "github", "figma"]}}]},
            # Deadline phrases that spaCy does not tag as DATE
            {"label": "DEADLINE", "pattern": [
                {"LOWER": "end"}, {"LOWER": "of"}, {"LOWER": "the"},
                {"LOWER": {"IN": ["day", "week", "sprint", "month"]}}]},
            {"label": "DEADLINE", "pattern": [
                {"LOWER": "next"},
                {"LOWER": {"IN": ["week", "sprint", "month", "release"]}}]},
            {"label": "DEADLINE", "pattern": [
                {"LOWER": "before"}, {"LOWER": "the"},
                {"LOWER": {"IN": ["release", "demo", "deadline"]}}]},
        ])
        _ner_nlp = nlp
    return _ner_nlp

def extract_entities(sentences):
    """Run NER on each sentence. Returns a list of dicts."""
    nlp = get_ner_nlp()
    results = []
    for i, doc in enumerate(nlp.pipe(sentences)):
        for ent in doc.ents:
            if ent.label_ in WANTED_LABELS:
                results.append({
                    "text": ent.text.strip(),
                    "label": ent.label_,
                    "sentence_id": i,
                    "sentence": sentences[i],
                })
    return results

def group_entities(entities):
    grouped = {}
    for e in entities:
        grouped.setdefault(e["label"], Counter())[e["text"]] += 1
    return grouped

def get_participants(entities, min_mentions=1):
    """People mentioned in the meeting (filtered and de-duplicated)."""
    names = Counter()
    for e in entities:
        if e["label"] != "PERSON":
            continue
        raw = re.sub(r"[^A-Za-z\s\.'-]", "", e["text"]).strip()
        raw = re.sub(TITLES, "", raw, flags=re.IGNORECASE)
        if len(raw) < 2 or not raw[0].isupper():      # real names are capitalized
            continue
        if raw.lower() in NOT_NAMES:
            continue
        names[raw.title()] += 1

    # Merge "Sarah" into "Sarah Johnson" when both appear
    merged = Counter()
    for name, count in names.items():
        longer = [n for n in names if n != name and name in n.split()]
        merged[longer[0] if longer else name] += count

    return [n for n, c in merged.most_common() if c >= min_mentions]

if __name__ == "__main__":
    import sys
    path = sys.argv[1]   # a *_sentences.txt file
    with open(path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    ents = extract_entities(sentences)
    grouped = group_entities(ents)

    print(f"Total entities found: {len(ents)}\n")
    for label, counter in grouped.items():
        print(f"{label}:")
        for text, count in counter.most_common(15):
            print(f"   {text}  (x{count})")
        print()

    print("Participants:", get_participants(ents))