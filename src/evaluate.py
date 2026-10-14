import os
import re
import sys
import csv
import glob
import difflib

import jiwer
from rouge_score import rouge_scorer
from bert_score import score as bert_score

TRANSCRIPT_DIR = "data/transcripts"

# ---------- helpers ----------
def read(path):
    with open(path, encoding="utf-8-sig") as f:
        return f.read().strip()

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

def norm(s):
    return re.sub(r"[^a-z0-9' ]", "", s.lower()).strip()

# ---------- summary metrics ----------
def rouge(pred, ref):
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r = sc.score(ref, pred)
    return {k: v.fmeasure for k, v in r.items()}

def bertscore_f1(preds, refs):
    _, _, f = bert_score(preds, refs, lang="en",
                         model_type="distilbert-base-uncased", verbose=False)
    return [x.item() for x in f]

def extractive_fraction(summary, transcript_sentences, threshold=0.85):
    """Share of summary sentences that appear almost verbatim in the transcript."""
    sents = split_sentences(summary)
    if not sents:
        return 0.0
    refs = [norm(t) for t in transcript_sentences]
    copied = 0
    for s in sents:
        ns = norm(s)
        best = max((difflib.SequenceMatcher(None, ns, r).ratio() for r in refs), default=0)
        if best >= threshold:
            copied += 1
    return copied / len(sents)

def eval_summaries(stem, reference_path):
    ref = read(reference_path)
    transcript = [l.strip() for l in open(
        os.path.join(TRANSCRIPT_DIR, f"{stem}_sentences.txt"), encoding="utf-8-sig") if l.strip()]
    total_words = sum(len(t.split()) for t in transcript)

    files = sorted(glob.glob(os.path.join(TRANSCRIPT_DIR, f"{stem}_summary_*.txt")))
    names = [os.path.basename(f)[len(stem) + len("_summary_"):-4] for f in files]
    preds = [read(f) for f in files]
    bert = bertscore_f1(preds, [ref] * len(preds))

    rows = []
    for name, pred, b in zip(names, preds, bert):
        r = rouge(pred, ref)
        rows.append({
            "model": name,
            "words": len(pred.split()),
            "compression": round(len(pred.split()) / total_words, 3),
            "rouge1": round(r["rouge1"], 3),
            "rouge2": round(r["rouge2"], 3),
            "rougeL": round(r["rougeL"], 3),
            "bertscore_f1": round(b, 3),
            "extractive_%": round(100 * extractive_fraction(pred, transcript)),
        })

    cols = list(rows[0].keys())
    print("  ".join(f"{c:<14}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):<14}" for c in cols))

    os.makedirs("data/outputs", exist_ok=True)
    out = os.path.join("data/outputs", f"{stem}_summary_eval.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("\nSaved to", out)

# ---------- transcription (WER) ----------
def segment_text(ts_path, start, end):
    """Text of all segments that start between `start` and `end` seconds."""
    out = []
    for line in open(ts_path, encoding="utf-8-sig"):
        m = re.match(r"\[(\d+(?:\.\d+)?) - (\d+(?:\.\d+)?)\]\s*(.*)", line)
        if m and start <= float(m.group(1)) < end:
            out.append(m.group(3).strip())
    return " ".join(out)

def clean_for_wer(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", t.lower())).strip()

# ---------- command line ----------
if __name__ == "__main__":
    cmd = sys.argv[1]

    if cmd == "summaries":            # summaries <stem> <reference_file>
        eval_summaries(sys.argv[2], sys.argv[3])

    elif cmd == "segment":            # segment <timestamps_file> <start> <end> <out_file>
        text = segment_text(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
        os.makedirs(os.path.dirname(sys.argv[5]) or ".", exist_ok=True)
        with open(sys.argv[5], "w", encoding="utf-8") as f:
            f.write(text)
        print("Draft saved to", sys.argv[5], f"({len(text.split())} words)")

    elif cmd == "wer":                # wer <reference_file> <timestamps_file> <start> <end>
        ref = clean_for_wer(read(sys.argv[2]))
        hyp = clean_for_wer(segment_text(sys.argv[3], float(sys.argv[4]), float(sys.argv[5])))
        print(f"WER: {100 * jiwer.wer(ref, hyp):.1f}%   ({len(ref.split())} reference words)")

    elif cmd == "diff":               # diff <reference_file> <timestamps_file> <start> <end>
        ref = clean_for_wer(read(sys.argv[2])).split()
        hyp = clean_for_wer(segment_text(sys.argv[3], float(sys.argv[4]), float(sys.argv[5]))).split()
        sm = difflib.SequenceMatcher(None, ref, hyp)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                print(f"{tag:8} reference: {' '.join(ref[i1:i2]) or '-':30} | model: {' '.join(hyp[j1:j2]) or '-'}")

    else:
        print("Unknown command. Use: summaries | segment | wer | diff")