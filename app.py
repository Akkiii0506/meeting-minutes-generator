import io
import os
import json
from datetime import datetime, time as dtime

import pandas as pd
import streamlit as st

from src.pipeline import generate_minutes, MinutesError
from src.export import export_docx, export_pdf

st.set_page_config(page_title="Meeting Minutes Generator", layout="wide")

UPLOAD_DIR = "data/uploads"
LARGE_FILE_MB = 150
ASR_OPTIONS = {"base (fast)": "base", "small (more accurate)": "small"}
SUMMARIZERS = {
    "BART-SAMSUM (best for dialogue)": "bart-samsum",
    "DistilBART": "distilbart",
    "FLAN-T5-base": "flan-t5-base",
    "FLAN-T5-large": "flan-t5-large",
    "No summary": None,
}
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# ---------- helpers ----------
def to_bytes(export_fn, minutes):
    buf = io.BytesIO()
    export_fn(minutes, buf)
    return buf.getvalue()

def clean_str(x):
    return x if isinstance(x, str) else ""

def lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]

def split_commas(text):
    return [p.strip() for p in text.split(",") if p.strip()]

def action_table(m):
    rows = []
    for a in m["action_items"]:
        dl = a["deadline"] or ""
        if dl and a.get("deadline_date"):
            dl = f"{dl} ({a['deadline_date']})"
        rows.append({"Task": a["task"], "Owner": ", ".join(a["owners"]), "Deadline": dl})
    return pd.DataFrame(rows, columns=["Task", "Owner", "Deadline"])

# ---------- header ----------
st.title("Meeting Minutes Generator")
st.caption("Upload a meeting recording or transcript and get structured minutes: "
           "summary, decisions, action items and more. Works best with clear, "
           "English-language recordings.")

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    asr_label = st.selectbox("Speech recognition model", list(ASR_OPTIONS), index=1)
    sum_label = st.selectbox("Summarization model", list(SUMMARIZERS), index=0)
    neutral_in = st.checkbox("Write 'the speaker' instead of 'I' in the summary", value=True,
                             help="The recording has no speaker labels. This stops the summary "
                                  "from crediting the speaker's tasks to a named person.")
    title_in = st.text_input("Meeting title (optional)")
    date_in = st.date_input("Meeting date", value=datetime.now().date())
    people_in = st.text_input("Participants (optional, comma-separated)",
                              help="Leave empty to detect names automatically. "
                                   "Names given here also help the speech model spell them correctly.")
    terms_in = st.text_input("Key terms (optional, comma-separated)",
                             help="Words the speech model often gets wrong, "
                                  "e.g. login, API, product names.")
    st.caption("The first run of each model downloads it, which can take a few minutes.")

# ---------- upload and run ----------
uploaded = st.file_uploader("Meeting recording or transcript",
                            type=["mp3", "wav", "mp4", "m4a", "webm", "txt"])

if uploaded is not None and uploaded.size > LARGE_FILE_MB * 1024 * 1024:
    st.warning(f"This file is larger than {LARGE_FILE_MB} MB. Processing may take a long time.")

run = st.button("Generate minutes", type="primary", disabled=uploaded is None)

if run and uploaded is not None:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, os.path.basename(uploaded.name))
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    participants = split_commas(people_in) or None
    terms = split_commas(terms_in) or None
    meeting_date = datetime.combine(date_in, dtime.min)

    status = st.status("Generating minutes...", expanded=True)
    try:
        minutes = generate_minutes(
            path, ASR_OPTIONS[asr_label], SUMMARIZERS[sum_label],
            title_in.strip() or None, meeting_date, participants,
            progress=status.write,
            terms=terms,
            neutral_speaker=neutral_in,
        )
        status.update(label="Minutes ready", state="complete", expanded=False)
        st.session_state["minutes"] = minutes
        st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1
    except MinutesError as e:
        status.update(label="Could not create the minutes", state="error", expanded=False)
        st.session_state.pop("minutes", None)
        st.error(str(e))
    except Exception as e:
        status.update(label="Something unexpected went wrong", state="error", expanded=False)
        st.session_state.pop("minutes", None)
        st.error("Something unexpected went wrong while creating the minutes. "
                 "Please try again, or try a different file.")
        with st.expander("Technical details"):
            st.exception(e)

# ---------- review and edit ----------
m = st.session_state.get("minutes")
if m:
    rid = st.session_state["run_id"]
    st.divider()
    st.subheader("Review and edit")
    for w in m.get("warnings", []):
        st.warning(w)
    st.info("This is an AI-generated draft. Check names, tasks and the summary before "
            "sharing. 'Speaker' means the person saying 'I' (the transcript has no "
            "speaker labels).")

    title = st.text_input("Title", m["title"], key=f"title_{rid}")

    c1, c2 = st.columns(2)
    with c1:
        people_text = st.text_area("Participants (one per line)",
                                   "\n".join(m["participants"]),
                                   key=f"people_{rid}", height=120)
        agenda_text = st.text_area("Agenda / topics (one per line)",
                                   "\n".join(m["agenda"]),
                                   key=f"agenda_{rid}", height=160)
    with c2:
        decisions_text = st.text_area("Key decisions (one per line)",
                                      "\n".join(m["key_decisions"]),
                                      key=f"dec_{rid}", height=120)
        st.markdown("**Next meeting**")
        if m["next_meeting"]:
            for n in m["next_meeting"]:
                when = f" ({n['when']})" if n.get("when") else ""
                st.markdown(f"- {n['sentence']}{when}")
        else:
            st.markdown("_not mentioned_")

    summary_text = st.text_area("Discussion summary", m["summary"] or "",
                                key=f"summary_{rid}", height=220)

    st.markdown("**Action items** (edit cells, add rows, or delete wrong ones)")
    edited = st.data_editor(action_table(m), num_rows="dynamic", key=f"actions_{rid}")

    # Build the final version from the edited fields
    final = dict(m)
    final["title"] = title
    final["participants"] = lines(people_text)
    final["agenda"] = lines(agenda_text)
    final["key_decisions"] = lines(decisions_text)
    final["summary"] = summary_text.strip()
    actions = []
    for _, r in edited.iterrows():
        task = clean_str(r["Task"]).strip()
        if not task:
            continue
        actions.append({
            "task": task,
            "owners": split_commas(clean_str(r["Owner"])),
            "deadline": clean_str(r["Deadline"]).strip() or None,
            "deadline_date": None,
            "confidence": "",
        })
    final["action_items"] = actions

    # ---------- downloads ----------
    st.subheader("Download")
    stem = os.path.splitext(m["source_file"])[0]
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Word (DOCX)", to_bytes(export_docx, final),
                       file_name=f"{stem}_minutes.docx", mime=DOCX_MIME)
    d2.download_button("⬇ PDF", to_bytes(export_pdf, final),
                       file_name=f"{stem}_minutes.pdf", mime="application/pdf")
    d3.download_button("⬇ JSON",
                       json.dumps(final, indent=2, ensure_ascii=False).encode("utf-8"),
                       file_name=f"{stem}_minutes.json", mime="application/json")

    st.caption(f"Processing time (seconds): {m['timings_seconds']}  |  "
               f"Summarizer: {m['models']['summarizer'] or 'none'}")