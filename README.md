# Meeting Minutes Generator

Turns a meeting recording (MP3, WAV, MP4, M4A, WEBM) or a transcript into structured minutes:
participants, agenda, discussion summary, key decisions, action items (owner and deadline),
and next meeting. The minutes can be reviewed and edited in a web app and downloaded as DOCX,
PDF, or JSON.

Built with Whisper (speech recognition), spaCy (named entity recognition), KeyBERT (keywords and
topics), and transformer summarizers (BART, FLAN-T5, PEGASUS). Everything runs locally on a CPU,
so meeting audio never leaves the computer (models are downloaded once on first use).

## Features
- Speech-to-text with `faster-whisper` (base or small model), with optional hints for participant names and key terms
- Transcript cleaning: filler words, repeats, spacing, sentence splitting
- Participants, action items (owner and deadline), decisions, keywords, topics, next meeting
- Abstractive, chunk-wise summary with a dialogue-trained BART model (BART-SAMSUM)
- Web app (Streamlit) with progress display, editable results, and friendly error messages
- Export to DOCX, PDF, and JSON

## How it works
```
audio/video -> ffmpeg -> Whisper ASR -> transcript
    -> cleaning + sentence splitting
    -> NER (participants), rules (action items, decisions, next meeting), KeyBERT (topics)
    -> chunk-wise transformer summary
    -> structured minutes -> DOCX / PDF / JSON
```

## Results on the test meeting
Test data: one 9-minute sprint meeting (public YouTube video, English, clear audio).
All times are on a laptop CPU (Lenovo IdeaPad Slim 5i, 16 GB RAM) with models already downloaded.

**Summarization models** (scored against a manually checked reference summary):

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | Sentences copied from transcript | Time |
|---|---|---|---|---|---|---|
| BART-SAMSUM | 0.578 | 0.261 | 0.305 | 0.840 | 21% | 32.5 s |
| BART-SAMSUM + "the speaker" rewrite | 0.538 | 0.257 | 0.321 | 0.837 | 31% | 36.1 s |
| DistilBART | 0.354 | 0.134 | 0.184 | 0.796 | 92% | 21.0 s |
| FLAN-T5-base | 0.333 | 0.120 | 0.190 | 0.781 | 80% | 21.5 s |
| FLAN-T5-large | 0.270 | 0.105 | 0.174 | 0.779 | 71% | 50.5 s |

The dialogue-tuned BART-SAMSUM was clearly best. News-trained and general models mostly copied
transcript sentences, and PEGASUS-XSum (trained on BBC articles) invented a news story.

**End to end:** about 288 s for a 9 min 10 s recording (transcription 233 s, extraction 13 s, summary 41 s).

## Installation (Windows, PowerShell)
1. Install Python 3.10 or 3.11 and Git.
2. Install **ffmpeg** (https://www.gyan.dev/ffmpeg/builds/), add its `bin` folder to PATH, and check with `ffmpeg -version`.
3. Set up the project:
```powershell
git clone <your-repository-url>
cd meeting-minutes-generator
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
The first run of each model downloads it (Whisper small about 0.5 GB, BART-SAMSUM about 1.6 GB).

## Usage
Web app:
```powershell
streamlit run app.py
```
Command line:
```powershell
python -m src.pipeline data/audio/meeting1.mp3 --asr small --participants "George, Wan" --terms "login, authentication, UI" --summarizer bart-samsum
```
Options: `--summarizer` (bart-samsum, distilbart, flan-t5-base, flan-t5-large, none), `--no-neutral` (keep "I" in the summary), `--title`, `--date YYYY-MM-DD`.

## Project structure
```
app.py                 Streamlit web app
src/
  transcribe.py        ffmpeg + faster-whisper
  preprocess.py        cleaning and sentence splitting
  extract.py           NER and participants
  actions.py           action items, owners, deadlines
  insights.py          decisions, keywords, topics
  summarize.py         chunk-wise summarization (several models)
  pipeline.py          combines everything into the minutes
  export.py            DOCX and PDF export
  evaluate.py          ROUGE, BERTScore, WER, copy rate
data/transcripts/      transcripts, sentence files, model summaries
data/eval/             reference text for evaluation
samples/               example output files
```

## Evaluation commands
```powershell
python -m src.evaluate summaries meeting1_small data/transcripts/meeting1_reference.txt
python -m src.evaluate wer data/eval/ref_2to4min.txt data/transcripts/meeting1_small_timestamps.txt 120 240
python -m src.evaluate diff data/eval/ref_2to4min.txt data/transcripts/meeting1_base_timestamps.txt 120 240
```

## Limitations
- Evaluated on one clean English meeting, so the numbers are indicative, not statistically strong.
- The transcript has no speaker labels: first-person statements appear as "Speaker", and the summary says "the speaker".
- Whisper can misspell names and domain words; participant names, key terms, and a glossary correction reduce this, but do not remove it.
- Action items, decisions, and next meeting are found with rules, so unusual phrasing is missed and vague "let's..." sentences can be included.
- A summary can still contain errors, so the app asks the user to review and edit the draft.
- English only. CPU processing takes roughly half the recording length.

## Future work
- Speaker identification (diarization) so tasks and the summary can name the real speaker
- Fine-tune the summarizer on meeting data (AMI, QMSum)
- Test on noisy, multi-speaker, and longer recordings
- Replace rule-based extraction with a trained or LLM-based model
- Multilingual support

## Data and ethics
The sample meeting comes from a public YouTube video (https://youtu.be/STlkuUwhK88?si=LKUO2hOOqSifScE2) and is used only
for academic testing. Audio files are not included in this repository. Meeting recordings can contain
personal information; this tool processes them locally.
