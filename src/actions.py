import re
import json
import dateparser
from datetime import datetime
from src.extract import get_ner_nlp, NOT_NAMES, TITLES

# Phrases that signal a commitment or request
TRIGGER = re.compile(
    r"\bwill\b|\b(?:i|we|you|they|he|she)[’']ll\b|\bgoing to\b|\bneed(?:s)? to\b|"
    r"\b(?:have|has|got) to\b|\bshould\b|\bmust\b|\blet[’']s\b|\blet us\b|"
    r"\b(?:can|could|would) you\b|\bplease\b|\bmake sure\b|\bresponsible for\b|"
    r"\bin charge of\b|\btake care of\b|\baction item\b",
    re.IGNORECASE,
)
REQUEST = re.compile(r"\b(?:can|could|would) you\b|\bplease\b", re.IGNORECASE)
DEADLINE_LABELS = {"DATE", "TIME", "DEADLINE"}

# Vague team talk that is not a concrete task
VAGUE = re.compile(
    r"\b(ensure|aim|focus|strategi[sz]e|remain|maintain|dive|monitor|"
    r"success(?:ful)?|smooth(?:ly)?|discuss|weigh|streamline)\b",
    re.IGNORECASE,
)

ROLE_PREFIX = r"^(?:tester|developer|dev|qa|manager|engineer|designer)\s+"

def is_action(doc):
    text = doc.text
    if len(text.split()) < 4:
        return False
    if not TRIGGER.search(text):
        return False
    if not any(t.pos_ == "VERB" for t in doc):          # needs a real verb
        return False
    if text.rstrip().endswith("?") and not REQUEST.search(text):
        return False                                     # plain questions are not tasks
    return True

def find_owners(doc):
    owners = []
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        name = re.sub(r"[^A-Za-z\s\.'-]", "", ent.text).strip()
        name = re.sub(TITLES, "", name, flags=re.IGNORECASE)
        name = re.sub(r"\s+", " ", name).strip()
        name = re.sub(ROLE_PREFIX, "", name, flags=re.IGNORECASE)   # "Tester George" -> "George"
        if len(name) > 1 and name[0].isupper() and name.lower() not in NOT_NAMES:
            owners.append(name.title())
    low = doc.text.lower()
    if re.search(r"\bi\b", low):
        owners.append("Speaker")
    if re.search(r"\bwe\b|\blet[’']s\b|\blet us\b", low):
        owners.append("Team")
    if not owners and re.search(r"\byou\b", low):
        owners.append("Addressee")
    return list(dict.fromkeys(owners)) or ["Unassigned"]   # remove duplicates, keep order

def find_deadlines(doc, base_date):
    out = []
    for ent in doc.ents:
        if ent.label_ in DEADLINE_LABELS:
            parsed = dateparser.parse(
                ent.text,
                settings={"RELATIVE_BASE": base_date, "PREFER_DATES_FROM": "future"},
            )
            if parsed and parsed.date() < base_date.date():
                continue                      # "yesterday" is not a deadline
            out.append({
                "text": ent.text,
                "date": parsed.strftime("%Y-%m-%d") if parsed else None,
            })
    return out

def classify(doc, owners, deadlines):
    specific = [o for o in owners if o not in ("Team", "Unassigned")]
    if specific or deadlines:
        return "high"
    if owners == ["Unassigned"]:
        return None                           # nobody is responsible
    if VAGUE.search(doc.text):
        return None                           # vague team talk
    return "medium"

def extract_action_items(sentences, meeting_date=None):
    nlp = get_ner_nlp()
    base = meeting_date or datetime.now()
    items = []
    for i, doc in enumerate(nlp.pipe(sentences)):
        if not is_action(doc):
            continue
        owners = find_owners(doc)
        deadlines = find_deadlines(doc, base)
        level = classify(doc, owners, deadlines)
        if level is None:
            continue
        items.append({
            "sentence_id": i,
            "task": doc.text.strip(),
            "owners": owners,
            "deadlines": deadlines,
            "confidence": level,
        })
    return items

if __name__ == "__main__":
    import sys, os
    path = sys.argv[1]   # a *_sentences.txt file
    with open(path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    items = extract_action_items(sentences)
    print(f"Action items found: {len(items)}\n")
    for n, it in enumerate(items, 1):
        dl = ", ".join(f"{d['text']} ({d['date']})" for d in it["deadlines"]) or "none"
        print(f"{n}. {it['task']}")
        print(f"   Owner: {', '.join(it['owners'])} | Deadline: {dl} | {it['confidence']}\n")

    out_path = path.replace("_sentences.txt", "_actions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print("Saved to", out_path)