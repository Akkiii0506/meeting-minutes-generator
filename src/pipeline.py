import os
import re
import json
import time
import argparse
from datetime import datetime

from src import transcribe as T
from src import summarize as S
from src.preprocess import preprocess
from src.extract import extract_entities, get_participants, get_ner_nlp
from src.actions import extract_action_items, find_deadlines
from src.insights import extract_decisions, extract_keywords

CACHE_DIR = "data/cache"
OUTPUT_DIR = "data/outputs"

NEXT_MEETING = re.compile(
    r"\b(reconvene|next meeting|meet again|follow[- ]up meeting|review meeting|"
    r"regroup|sync up|catch up)\b", re.IGNORECASE)

ACRONYMS = {"ui", "ux", "api", "qa", "ci", "cd", "sdk"}

def smart_title(text):
    """Title-case a phrase but keep acronyms like UI in capitals."""
    return " ".join(w.upper() if w.lower() in ACRONYMS else w.capitalize()
                    for w in text.split())

# ---------- caching ----------
def _cache_path(path, tag):
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(CACHE_DIR, f"{stem}_{tag}.json")

def _load_cache(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    return None

def _save_cache(path, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------- next meeting ----------
def find_next_meeting(sentences, base_date):
    nlp = get_ner_nlp()
    found = []
    for s in sentences:
        if NEXT_MEETING.search(s) and not s.rstrip().endswith("?"):
            dl = find_deadlines(nlp(s), base_date)
            found.append({
                "sentence": s,
                "when": dl[0]["text"] if dl else None,
                "date": dl[0]["date"] if dl else None,
            })
    return found

# ---------- main pipeline ----------
def generate_minutes(path, asr_size="small", summarizer="flan-t5-base",
                     title=None, meeting_date=None, participants=None,
                     progress=print):
    timings = {}
    stem = os.path.splitext(os.path.basename(path))[0]
    meeting_date = meeting_date or datetime.now()

    # 1. Transcript
    t0 = time.time()
    if path.lower().endswith(".txt"):
        with open(path, encoding="utf-8-sig") as f:
            raw = f.read()
    else:
        cp = _cache_path(path, f"transcript_{asr_size}")
        cached = _load_cache(cp)
        if cached:
            raw = cached["text"]
        else:
            progress("Transcribing audio...")
            raw, _ = T.transcribe(path, asr_size, save_dir=CACHE_DIR)
            _save_cache(cp, {"text": raw})
    timings["transcription"] = round(time.time() - t0, 1)

    # 2. Cleaning
    progress("Cleaning transcript...")
    t0 = time.time()
    sentences = preprocess(raw)["sentences"]
    timings["preprocessing"] = round(time.time() - t0, 1)

    # 3. Information extraction
    progress("Extracting participants, action items, decisions, topics...")
    t0 = time.time()
    found_people = get_participants(extract_entities(sentences))
    people = participants or found_people
    actions = extract_action_items(sentences, meeting_date)
    decisions = extract_decisions(sentences)
    keywords, topics = extract_keywords(sentences)
    next_meeting = find_next_meeting(sentences, meeting_date)

    # Show each item in only one section: remove action items that are
    # already listed as a decision or as the next meeting
    skip = {d["decision"] for d in decisions} | {n["sentence"] for n in next_meeting}
    actions = [a for a in actions if a["task"] not in skip]
    timings["extraction"] = round(time.time() - t0, 1)

    # 4. Summary
    summary = None
    t0 = time.time()
    if summarizer:
        sp = _cache_path(path, f"summary_{summarizer}")
        cached = _load_cache(sp)
        if cached:
            summary = cached["summary"]
        else:
            progress(f"Summarizing with {summarizer} (can take several minutes on CPU)...")
            res = S.summarize_meeting(sentences, summarizer)
            summary = res["final"]
            _save_cache(sp, {"summary": summary, "chunks": res["partials"]})
    timings["summarization"] = round(time.time() - t0, 1)

    minutes = {
        "title": title or f"Meeting Minutes - {stem}",
        "date": meeting_date.strftime("%Y-%m-%d"),
        "source_file": os.path.basename(path),
        "participants": people,
        "agenda": [smart_title(t) for t in topics],
        "summary": summary,
        "key_decisions": [d["decision"] for d in decisions],
        "action_items": [{
            "task": a["task"],
            "owners": a["owners"],
            "deadline": a["deadlines"][0]["text"] if a["deadlines"] else None,
            "deadline_date": a["deadlines"][0]["date"] if a["deadlines"] else None,
            "confidence": a["confidence"],
        } for a in actions],
        "keywords": keywords,
        "next_meeting": next_meeting,
        "models": {"asr": asr_size if not path.lower().endswith(".txt") else None,
                   "summarizer": summarizer},
        "timings_seconds": timings,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{stem}_minutes.json"), "w", encoding="utf-8") as f:
        json.dump(minutes, f, indent=2, ensure_ascii=False)
    return minutes

# ---------- readable text version ----------
def format_minutes_text(m):
    lines = [m["title"].upper(), f"Date: {m['date']}", ""]
    lines += ["PARTICIPANTS"] + [f"  - {p}" for p in m["participants"]] + [""]
    lines += ["AGENDA / TOPICS"] + [f"  {i}. {t}" for i, t in enumerate(m["agenda"], 1)] + [""]
    lines += ["DISCUSSION SUMMARY", f"  {m['summary'] or '(no summary)'}", ""]
    lines += ["KEY DECISIONS"] + ([f"  - {d}" for d in m["key_decisions"]] or ["  (none found)"]) + [""]
    lines += ["ACTION ITEMS"]
    for i, a in enumerate(m["action_items"], 1):
        dl = a["deadline"] or "no deadline"
        lines.append(f"  {i}. {a['task']}")
        lines.append(f"     Owner: {', '.join(a['owners'])} | Deadline: {dl}")
    lines += [""]
    lines += ["NEXT MEETING"]
    for n in m["next_meeting"] or [{"sentence": "(not mentioned)", "when": None}]:
        lines.append(f"  - {n['sentence']}" + (f" ({n['when']})" if n.get("when") else ""))
    return "\n".join(lines)

# ---------- command line ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="audio/video file or .txt transcript")
    ap.add_argument("--asr", default="small")
    ap.add_argument("--summarizer", default="flan-t5-base",
                    help="flan-t5-base, distilbart, bart-samsum, ... or 'none'")
    ap.add_argument("--title")
    ap.add_argument("--date", help="YYYY-MM-DD")
    ap.add_argument("--participants", help="comma-separated names")
    a = ap.parse_args()

    result = generate_minutes(
        a.path, a.asr,
        None if a.summarizer.lower() == "none" else a.summarizer,
        a.title,
        datetime.strptime(a.date, "%Y-%m-%d") if a.date else None,
        [p.strip() for p in a.participants.split(",")] if a.participants else None,
    )
    print("\n" + "=" * 60)
    print(format_minutes_text(result))
    print("=" * 60)
    print("Timings (s):", result["timings_seconds"])