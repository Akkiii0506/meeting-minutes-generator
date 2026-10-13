import faster_whisper, spacy, transformers, torch, docx, streamlit
print("All imports OK")
print("spaCy model:", spacy.load("en_core_web_sm").meta["name"])
print("CUDA available:", torch.cuda.is_available())  # expect False on your laptop