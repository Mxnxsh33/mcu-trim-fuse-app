# MCU Trim-Code Predictor, v4 (explained for everyone)

Predicts the best bandgap, HSI-oscillator and LDO trim codes of a microcontroller die from pre-trim measurements,
builds and validates the 64-bit fuse word, and explains everything in plain language.

## What is new in v4
- **📖 Start here tab:** a plain-language tour (6 steps, a "words in plain English" table, "try it in three clicks", honest limits).
- **🟢 Plain words tab** inside "Do you know what you just did?": your chip's result explained with everyday comparisons.
- **Simple / Detailed switch** in the Project assistant. Simple (default) answers in everyday words from `simple_guide.md`;
  Detailed uses the technical notes in `project_knowledge.md`.
- Better offline search (word + letter-pattern matching + synonyms) so rephrased questions still find the right answer.

## Run locally (Windows, from D:)
    set TEMP=D:\mcu\tmp
    set TMP=D:\mcu\tmp
    set PIP_CACHE_DIR=D:\mcu\pipcache
    pip install -r requirements.txt
    python -m streamlit run app.py

## LLM mode (optional)
Paste a key in the *Project assistant* tab, or set `ANTHROPIC_API_KEY`, or copy `.streamlit/secrets.toml.example` to `secrets.toml`.
Never commit a key. 25 LLM questions per session, then offline mode. LLM mode has not been tested with a real key.

## Files
- `app.py` UI | `explain.py` plain-English explainer | `assistant.py` LLM + offline assistant | `fuse_tools.py` fuse build/validate
- `simple_guide.md` beginner notes | `project_knowledge.md` technical notes | `mcu_trim_utils.py` shared code + `TrimPredictor`
- `models/` saved model, model card, app config, transfer results | `sample_dies.csv`, `sample_fuse_words.csv` demo data
- `experiments/` reproducible scripts
