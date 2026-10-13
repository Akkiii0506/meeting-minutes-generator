Problem statement: manual minute-taking is slow and error-prone
Input: MP3/WAV/MP4 meeting recording (and optionally a ready transcript)
Output: structured minutes as PDF/DOCX
Core features: transcription, cleaning, NER, action items and deadlines, decisions, abstractive summary, export
Optional extras: speaker identification, Streamlit UI polish
Models: faster-whisper small, spaCy, FLAN-T5 and BART (comparison on Colab)