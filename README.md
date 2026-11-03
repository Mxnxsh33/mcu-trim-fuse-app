# MCU Trim-Code Predictor, v3 (trim + fuse + explainer + LLM assistant)

Predicts the best bandgap, HSI-oscillator and LDO trim codes of a microcontroller die from pre-trim measurements
(Random Forest on top of a physics estimate), builds and validates the 64-bit fuse word, explains every prediction in plain
English and has a project assistant (LLM mode with an API key, offline mode without).

## What is new in v3
- **Fuse map tab:** build the fuse word for the current die; check any fuse word or a CSV against rule-based checks (`fuse_tools.py`).
  Caught 406/406 and 383/383 injected errors on two labelled files, 0 false alarms. No ML decides any fuse bit.
- **Transfer-learning result** (About tab, `experiments/`): the model works on a new chip type (MCU-C) with no MCU-C data.
- Project notes extended with sections 12 and 13.

## Run locally (Windows, from D:)
    set TEMP=D:\mcu\tmp
    set TMP=D:\mcu\tmp
    set PIP_CACHE_DIR=D:\mcu\pipcache
    pip install -r requirements.txt
    python -m streamlit run app.py

## LLM mode (optional)
Paste a key in the *Project assistant* tab, or set `ANTHROPIC_API_KEY`, or copy `.streamlit/secrets.toml.example` to `secrets.toml`.
Never commit a key. 25 LLM questions per session, then offline mode.

## Files
- `app.py` UI | `explain.py` plain-English explainer | `assistant.py` LLM + offline assistant | `fuse_tools.py` fuse build/validate
- `project_knowledge.md` notes the assistant is grounded on | `mcu_trim_utils.py` shared feature code + `TrimPredictor`
- `models/` saved model, model card, app config, transfer results | `sample_dies.csv`, `sample_fuse_words.csv` demo data
- `experiments/` reproducible scripts: `fuse_validation_check.py`, `transfer_learning_experiment.py`
