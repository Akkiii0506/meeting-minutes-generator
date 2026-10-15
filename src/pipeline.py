import os
import re
import json
import time
import difflib
import hashlib
import argparse
import subprocess
from datetime import datetime

from src import transcribe as T
from src import summarize as S
from src.preprocess import preprocess
from src.extract import extract_entities, get_participants, get_ner_nlp
from src.actions import extract_action_items, find_deadlines
from src.insights import extract_decisions, extract_keywords

CACHE_DIR = "data/cache"
OUTPUT_DIR = "data/outputs"
MIN_SENTENCES = 3

NEXT_MEETING = re.compile(
    r"\b(reconvene|next meeting|meet again|follow[- ]up meeting|review meeting|"
    r"regroup|sync up|catch up)\b", re.IGNORECASE)

ACRONYMS = {"ui", "ux", "api", "qa", "ci", "cd", "sdk"}

class MinutesError(Exception):
    """A problem with a message that is safe and useful to show to the user."""

def smart_title(text):
    """Title-case a phrase but keep acronyms like UI in capitals."""
    return " ".join(w.upper() if w.lower() in ACRONYMS else w.capitalize()
                    for w in text.split())

# ---------- glossary correction ----------
def apply_glossary(text, terms, protect=None):
    """Fix words the speech model got wrong, using the user's key terms.
    Only capitalised words in the middle of a sentence are considered, and
    participant names are never changed."""
    if not terms:
        return text
    protect_l = set()
    for p in protect or []:
        protect_l.add(p.lower())
        protect_l.update(p.lower().split())
    singles = [t for t in terms if " " not in t]

    out, last = [], 0
    for m in re.finditer(r"[A-Za-z]+", text):
        w = m.group()
        before = text[:m.start()].rstrip()
        mid_sentence = bool(before) and before[-1] not in ".!?"
        repl = None
        if mid_sentence and w[0].isupper() and len(w) > 3 and w.lower() not in protect_l:
            for t in singles:
                if t.lower() != w.lower() and \
                   difflib.SequenceMatcher(None, w.lower(), t.lower()).ratio() >= 0.75:
                    repl = t
                    break
        out.append(text[last:m.start()])
        out.append(repl if repl else w)
        last = m.end()
    out.append(text[last:])
    return "".join(out)

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
                     progress=print, terms=None):
    timings = {}
    warn_list = []
    stem = os.path.splitext(os.path.basename(path))[0]
    meeting_date = meeting_date or datetime.now()

    if not os.path.exists(path):
        raise MinutesError(f"The file '{os.path.basename(path)}' was not found.")

    # 1. Transcript
    t0 = time.time()
    if path.lower().endswith(".txt"):
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            raw = f.read()
    else:
        parts = []
        if participants:
            parts.append("Meeting participants: " + ", ".join(participants) + ".")
        if terms:
            parts.append("Terms: " + ", ".join(terms) + ".")
        hint = " ".join(parts) or None
        tag = f"transcript_{asr_size}"
        if hint:
            tag += "_" + hashlib.md5(hint.encode("utf-8")).hexdigest()[:6]
        cp = _cache_path(path, tag)
        cached = _load_cache(cp)
        if cached:
            raw = cached["text"]
        else:
            progress("Transcribing audio...")
            try:
                raw, _ = T.transcribe(path, asr_size, save_dir=CACHE_DIR, hint=hint)
            except FileNotFoundError:
                raise MinutesError(
                    "A required program was not found. Check that ffmpeg is installed "
                    "and added to PATH, then restart the app.") from None
            except subprocess.CalledProcessError:
                raise MinutesError(
                    "This file could not be read as audio or video. Try another file, "
                    "or convert it to MP3 or WAV first.") from None
            if raw.strip():                       # do not cache an empty transcript
                _save_cache(cp, {"text": raw})

    # Correct words the speech model got wrong, using the user's key terms
    raw = apply_glossary(raw, terms, protect=participants)
    timings["transcription"] = round(time.time() - t0, 1)

    # 2. Cleaning
    progress("Cleaning transcript...")
    t0 = time.time()
    sentences = preprocess(raw)["sentences"]
    timings["preprocessing"] = round(time.time() - t0, 1)

    if len(sentences) < MIN_SENTENCES:
        raise MinutesError(
            f"Not enough speech was found ({len(sentences)} sentence(s)). Please use a "
            "recording with clear spoken English and check that the file is not silent.")

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

    # 4. Summary (a failure here should not lose the rest of the minutes)
    summary = None
    t0 = time.time()
    if summarizer:
        try:
            sp = _cache_path(path, f"summary_{summarizer}")
            cached = _load_cache(sp)
            if cached:
                summary = cached["summary"]
            else:
                progress(f"Summarizing with {summarizer} (can take a minute on CPU)...")
                res = S.summarize_meeting(sentences, summarizer)
                summary = res["final"]
                _save_cache(sp, {"summary": summary, "chunks": res["partials"]})
        except Exception as e:
            summary = None
            warn_list.append(
                f"The summary could not be created ({type(e).__name__}). The rest of the "
                "minutes are ready. If this is the first use of this summarization model, "
                "check your internet connection (the model must be downloaded once).")
            progress("Summary skipped because of an error.")
    timings["summarization"] = round(time.time() - t0, 1)

    if not actions:
        warn_list.append("No action items were detected in this recording.")
    if not decisions:
        warn_list.append("No explicit decisions were detected in this recording.")

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
        "warnings": warn_list,
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
    for w in m.get("warnings", []):
        lines.append(f"\nNOTE: {w}")
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
    ap.add_argument("--terms", help="comma-separated key terms, e.g. login, API")
    a = ap.parse_args()

    try:
        result = generate_minutes(
            a.path, a.asr,
            None if a.summarizer.lower() == "none" else a.summarizer,
            a.title,
            datetime.strptime(a.date, "%Y-%m-%d") if a.date else None,
            [p.strip() for p in a.participants.split(",")] if a.participants else None,
            terms=[t.strip() for t in a.terms.split(",") if t.strip()] if a.terms else None,
        )
    except MinutesError as e:
        raise SystemExit(f"Error: {e}")

    print("\n" + "=" * 60)
    print(format_minutes_text(result))
    print("=" * 60)
    print("Timings (s):", result["timings_seconds"])