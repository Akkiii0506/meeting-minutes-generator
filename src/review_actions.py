import sys
import json
from src.extract import get_ner_nlp
from src.actions import is_action

sentences = [l.strip() for l in open(sys.argv[1], encoding="utf-8-sig") if l.strip()]
with open(sys.argv[2], encoding="utf-8-sig") as f:
    minutes = json.load(f)

kept = {a["task"] for a in minutes["action_items"]}
skip = set(minutes["key_decisions"]) | {n["sentence"] for n in minutes["next_meeting"]}

nlp = get_ner_nlp()
print("Sentences that look like tasks but are NOT in the action items:\n")
for s, doc in zip(sentences, nlp.pipe(sentences)):
    if is_action(doc) and s not in kept and s not in skip:
        print("-", s)