"""
assistant.py
============
The project chat assistant.

Two modes:
  * LLM mode     - a pretrained large language model (via the Anthropic API) answers, grounded on
                   project_knowledge.md plus the die currently shown in the app.
  * Offline mode - no API key / no internet / question limit reached: the most relevant section of
                   project_knowledge.md is found with TF-IDF text similarity and shown.

Nothing here changes the ML model or its predictions. The assistant only explains.
No LLM is trained from scratch; a pretrained one is used through an API.
"""
import os
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL = "claude-haiku-5-5"   # small/fast model; override with ANTHROPIC_MODEL
MAX_HISTORY_MESSAGES = 8             # how much of the chat is sent back to the LLM (keeps cost low)
MAX_USER_CHARS = 1000                # longest question accepted
MAX_LLM_QUESTIONS = 25               # per browser session; afterwards offline mode (protects your API budget)
MAX_OUTPUT_TOKENS = 700

SYSTEM_TEMPLATE = """You are the Project Assistant inside the "MCU Trim-Code Predictor" app, a machine-learning project that predicts microcontroller trim codes from pre-trim measurements.

RULES
- Answer using ONLY the PROJECT NOTES and CURRENT DIE below. If the answer is not there, say you do not know that from the project and suggest related topics the notes do cover.
- Explain in simple words first, then add detail if useful. Define jargon the first time you use it. The reader may be a student preparing to present this project in an interview.
- Never invent numbers. Quote results exactly as written in the notes. Always be honest about limitations (for example HSI accuracy drops on new lots).
- Stay on topic: this project, the chip-trimming concepts needed to understand it, and the machine-learning ideas it uses. Politely decline unrelated requests.
- You only explain. You cannot change the model, run predictions or access files.
- Keep answers short (under about 200 words) unless the user asks for more detail. Use short paragraphs or bullets.

=== PROJECT NOTES ===
{kb}

=== CURRENT DIE SHOWN IN THE APP ===
{ctx}
"""


# ----------------------------------------------------------------------------
# knowledge base + offline retrieval
# ----------------------------------------------------------------------------
def load_kb(path):
    return Path(path).read_text(encoding="utf-8")


def split_chunks(kb):
    """Split the notes into (title, text) chunks at '### ' headings (section intros kept too)."""
    chunks, title, buf, section = [], None, [], ""
    for line in kb.splitlines():
        if line.startswith("## "):
            if title and buf:
                chunks.append((title, "\n".join(buf).strip()))
            section, title, buf = line[3:].strip(), None, []
        elif line.startswith("### "):
            if title and buf:
                chunks.append((title, "\n".join(buf).strip()))
            title, buf = line[4:].strip(), []
        elif title is not None:
            buf.append(line)
    if title and buf:
        chunks.append((title, "\n".join(buf).strip()))
    return [(t, x) for t, x in chunks if x]


CURRENT_DIE_PATTERNS = re.compile(
    r"(my|this|current|the)\s+(prediction|die|result|output|codes?)|what did i (just )?do|explain (my|this|the) (prediction|result|output)",
    re.I)


def is_current_die_question(q):
    return bool(CURRENT_DIE_PATTERNS.search(q))


class OfflineAnswerer:
    """TF-IDF search over the notes. Simple, fast, no internet."""

    def __init__(self, kb):
        self.chunks = split_chunks(kb)
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        # headings are weighted by repeating them, so "why random forest" finds the right section
        self.matrix = self.vec.fit_transform([f"{t} {t} {t} {x}" for t, x in self.chunks])

    def answer(self, question, current_summary=None):
        footer = "\n\n_Offline mode: answered straight from the project notes. Connect an API key for free-form answers._"
        if current_summary and is_current_die_question(question):
            return current_summary + footer
        question = re.sub(r"\b(chat ?bots?|bots?|llms?)\b", "assistant llm", question, flags=re.I)   # users say chatbot, notes say assistant
        sims = cosine_similarity(self.vec.transform([question]), self.matrix).ravel()
        order = sims.argsort()[::-1]
        best = order[0]
        if sims[best] < 0.06:
            topics = ", ".join(f"*{self.chunks[i][0]}*" for i in range(0, len(self.chunks), max(1, len(self.chunks) // 5))[:5])
            return ("I could not find that in the project notes. Try asking about the model choice, the data, "
                    f"the results, the limitations or the interview questions (for example {topics})." + footer)
        parts = [f"**{self.chunks[best][0]}**\n\n{self.chunks[best][1]}"]
        second = order[1]
        if sims[second] >= 0.7 * sims[best] and sims[second] >= 0.1:
            parts.append(f"**{self.chunks[second][0]}**\n\n{self.chunks[second][1]}")
        return "\n\n---\n\n".join(parts) + footer


# ----------------------------------------------------------------------------
# LLM mode
# ----------------------------------------------------------------------------
def llm_available():
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def build_system_prompt(kb, current_context):
    return SYSTEM_TEMPLATE.format(kb=kb, ctx=current_context or "(no die entered yet)")


def clean_history(history):
    """Keep the last few messages, make sure it starts with a user message and ends with the new question."""
    msgs = [{"role": m["role"], "content": m["content"][:4000]} for m in history if m["role"] in ("user", "assistant")]
    msgs = msgs[-MAX_HISTORY_MESSAGES:]
    while msgs and msgs[0]["role"] != "user":
        msgs = msgs[1:]
    return msgs


def stream_llm_answer(api_key, model, system, history, offline_fallback=None, client_factory=None):
    """Generator of text pieces for st.write_stream. Never raises: errors become a friendly message
    (followed by the offline answer when a fallback function is given)."""
    try:
        import anthropic
        client = client_factory(api_key) if client_factory else anthropic.Anthropic(api_key=api_key, timeout=45.0, max_retries=1)
        with client.messages.stream(model=model, max_tokens=MAX_OUTPUT_TOKENS,
                                    system=system, messages=clean_history(history)) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:  # noqa: BLE001 - any failure must degrade gracefully
        name = type(e).__name__
        if "Authentication" in name or "PermissionDenied" in name:
            msg = "The API key was rejected. Check that it is correct and active."
        elif "RateLimit" in name:
            msg = "The API is rate-limiting requests right now. Try again in a moment."
        elif "Connection" in name or "Timeout" in name:
            msg = "Could not reach the LLM service (network problem)."
        elif "NotFound" in name:
            msg = "The configured model name was not found. Set ANTHROPIC_MODEL to a valid model."
        else:
            msg = f"The LLM request failed ({name})."
        yield f"⚠️ {msg} Showing the offline answer instead.\n\n"
        if offline_fallback:
            yield offline_fallback()


def get_api_key(session_key=None, secrets=None):
    """Key priority: pasted in the app > environment variable > Streamlit secrets."""
    if session_key:
        return session_key.strip()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"].strip()
    try:
        if secrets is not None and secrets.get("ANTHROPIC_API_KEY"):
            return str(secrets["ANTHROPIC_API_KEY"]).strip()
    except Exception:  # noqa: BLE001 - no secrets file is normal
        pass
    return None


def get_model(secrets=None):
    if os.environ.get("ANTHROPIC_MODEL"):
        return os.environ["ANTHROPIC_MODEL"]
    try:
        if secrets is not None and secrets.get("ANTHROPIC_MODEL"):
            return str(secrets["ANTHROPIC_MODEL"])
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_MODEL
