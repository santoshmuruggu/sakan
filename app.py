"""
app.py — Day 5, Steps 5.1-5.2 of the Sakan build plan.

The Streamlit chat UI: type a question about Dubai tenancy law, get an
answer grounded in the actual law text, with the source article(s) shown
so you can check it yourself.
"""

import os

import streamlit as st

# On Streamlit Community Cloud, the API key is set as a "Secret" (read via
# st.secrets), not a local .env file. st.secrets raises if no secrets.toml
# exists at all (true for local dev, which uses .env instead) so this has
# to be guarded rather than a plain "in" check. Bridging it into the same
# environment variable generate.py already reads means the exact same
# code works both locally and once deployed.
try:
    if "ANTHROPIC_API_KEY" in st.secrets and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except st.errors.StreamlitSecretNotFoundError:
    pass

from generate import REFUSAL, generate_answer
from retrieve import Retriever, chunk_label

st.set_page_config(page_title="Sakan — Dubai Tenancy Law Q&A", page_icon="🏠")

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

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error(
        "No ANTHROPIC_API_KEY configured. Add it to a local .env file, or "
        "as a Secret if this app is deployed on Streamlit Community Cloud."
    )
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
        with st.spinner("Searching the law and drafting an answer..."):
            chunks = retriever.hybrid_search(question, k=6)
            answer = generate_answer(question, chunks)
        st.markdown(answer)

        sources = [] if answer.strip().startswith(REFUSAL) else chunks
        if sources:
            render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
