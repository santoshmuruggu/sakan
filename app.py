"""
app.py — the Streamlit chat UI: type a question about Dubai tenancy law, get an
answer grounded in the actual law text, with the source article(s) shown
so you can check it yourself.
"""

import anthropic
import httpx
import streamlit as st

from generate import (
    OLLAMA_MODEL,
    PROVIDER,
    REFUSAL,
    generate_answer,
    local_anthropic_key,
)
from retrieve import Retriever, chunk_label

st.set_page_config(
    page_title="Sakan — Dubai Tenancy Law Q&A",
    page_icon="🏠",
    initial_sidebar_state="expanded",
)

st.title("🏠 Sakan")
st.caption("A grounded Q&A assistant for Dubai tenancy law")

st.warning(
    "**Informational demo, not legal advice.** Answers are generated only "
    "from a fixed set of source documents (Law 26/2007, Law 33/2008, "
    "Decree 43/2013, and the DLD Tenancy Guide) and may be incomplete or "
    "out of date. Always verify current figures and requirements with the "
    "[Dubai Land Department](https://dubailand.gov.ae) before relying on "
    "them.",
    icon="⚠️",
)

RUN_LOCALLY = """
**Don't have a Claude API key? Run Sakan locally for free with Ollama.**
Everything runs on your own machine: retrieval, embeddings and the model.

1. Install [Ollama](https://ollama.com/download) and pull a model:
   ```bash
   ollama pull llama3.1:8b
   ```
2. Clone and install Sakan:
   ```bash
   git clone https://github.com/santoshmuruggu/sakan.git
   cd sakan
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Tell Sakan to use Ollama, then start it:
   ```bash
   echo "LLM_PROVIDER=ollama" > .env
   streamlit run app.py
   ```

Any Ollama chat model works: set `OLLAMA_MODEL=<name>` in `.env` to pick
a different one. Small local models follow the citation and refusal rules
less reliably than Claude Haiku, which the eval numbers in the README were
measured with.
"""

# Which model answers, decided once per visitor:
#   - LLM_PROVIDER=ollama: a local model, no key needed (local runs only).
#   - otherwise Claude: a key from a local .env file if this is someone
#     running it on their own machine, or else the key the visitor pastes
#     in below. The live demo deliberately has no key of its own — any
#     platform-injected ANTHROPIC_API_KEY is ignored (see
#     generate.local_anthropic_key).
api_key = None
if PROVIDER == "ollama":
    st.info(f"Running locally with Ollama model `{OLLAMA_MODEL}`.", icon="🦙")
else:
    api_key = local_anthropic_key()
    if not api_key:
        with st.sidebar:
            st.subheader("🔑 Claude API key")
            entered = st.text_input(
                "Anthropic API key",
                type="password",
                placeholder="sk-ant-...",
                help="Get one at console.anthropic.com. It's held only in this "
                     "browser session, used only to answer your questions, and "
                     "never stored or logged.",
            )
            st.caption(
                "Each question is one Claude Haiku call: a fraction of a cent. "
                "[Get a key](https://console.anthropic.com/settings/keys)"
            )
        api_key = entered.strip() or None

    if not api_key:
        st.info(
            "**This demo needs a Claude API key to answer questions.** Paste "
            "yours in the sidebar (open it with **>** at the top left on "
            "mobile).",
            icon="🔑",
        )
        st.markdown(RUN_LOCALLY)
        st.stop()
    if not api_key.startswith("sk-ant-"):
        st.error("That doesn't look like an Anthropic API key. They start with `sk-ant-`.")
        st.stop()


@st.cache_resource(show_spinner="Loading the law corpus and search indexes...")
def get_retriever() -> Retriever:
    return Retriever()


retriever = get_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(chunks: list[dict]):
    with st.expander(f"Sources ({len(chunks)})"):
        for chunk in chunks:
            st.markdown(f"**{chunk_label(chunk)}**")
            st.markdown(chunk["text"])
            st.divider()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

question = st.chat_input("Ask a question about Dubai tenancy law...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the law and drafting an answer..."):
                chunks = retriever.hybrid_search(question, k=6)
                answer = generate_answer(question, chunks, api_key)
        except anthropic.AuthenticationError:
            st.session_state.messages.pop()
            st.error("Anthropic rejected that API key. Check it in the sidebar and try again.")
            st.stop()
        except anthropic.APIStatusError as exc:
            st.session_state.messages.pop()
            st.error(f"The Claude API returned an error ({exc.status_code}). Try again in a moment.")
            st.stop()
        except httpx.HTTPError as exc:
            st.session_state.messages.pop()
            st.error(
                f"Couldn't get an answer from Ollama at its local address ({exc}). "
                f"Is `ollama serve` running, and have you pulled `{OLLAMA_MODEL}`?"
            )
            st.stop()
        st.markdown(answer)

        sources = [] if answer.strip().startswith(REFUSAL) else chunks
        if sources:
            render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
