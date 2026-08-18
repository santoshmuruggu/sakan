# Sakan — Step-by-Step Build Plan

**Sakan** = a Q&A chatbot for Dubai rental law. You type a question like "How much
notice does my landlord need to give before raising rent?" and it answers using
the *actual law text* (not guesses), and shows you exactly which Article the
answer came from. If the law doesn't cover your question, it says so instead of
making something up.

This document is written so you can follow it **one step at a time**, even if
you've never built something like this before. Every step explains *what*
you're doing and *why*, before *how*. We will not skip ahead — each step
produces something you can actually run and check before moving to the next.

---

## Part 1 — Concepts you need before we start

You don't need to memorize this section. Read it once, then come back to it
whenever a term in a later step is unfamiliar.

### What is "RAG"?
RAG = **Retrieval-Augmented Generation**. Instead of asking an AI model a
question and letting it answer from memory (which is how it can "hallucinate" —
confidently make things up), we:
1. **Retrieve** the specific paragraphs of law that are relevant to the question.
2. **Feed those paragraphs to the AI** along with the question.
3. Tell the AI: "only answer using what I gave you, and say where you got it."

This is why Sakan won't lie about rent-increase percentages — we force it to
quote the actual law, not recite something it half-remembers from training.

### Key building blocks

| Term | Plain explanation |
|---|---|
| **Corpus** | The set of source documents we're building answers from — here, Dubai's rental law PDFs. |
| **Chunking** | Cutting a big document into smaller pieces (chunks) so we can search over them individually. We'll cut by *Article number*, not by a fixed character count, because splitting an article in half would break its meaning. |
| **Embedding** | Turning a piece of text into a list of numbers (a "vector") that represents its *meaning*. Similar meanings → similar numbers. This lets a computer find "related" text even if the exact words differ. |
| **Vector store** | A database built specifically to store embeddings and quickly find the ones most similar to a new query. We'll use **ChromaDB** (runs locally, free, no server setup). |
| **BM25 / keyword search** | A classic search method (like Ctrl+F but smarter) that matches on exact words. Good for legal terms like "Ejari" or "Article 25" that embeddings can sometimes blur together. |
| **Hybrid search** | Running *both* vector search and keyword search, then combining the results, so we get the strengths of each. |
| **Reciprocal Rank Fusion (RRF)** | The specific (simple) math formula used to merge two ranked result lists into one combined ranking. |
| **LLM (Large Language Model)** | The AI model that actually writes the final answer sentence, e.g. an OpenAI or Anthropic model, called through an API. |
| **Prompt** | The instructions we send to the LLM. Ours will say: "only use the text I give you, cite the article number, and if it's not in there, say you don't know." |
| **Citation verification** | After the LLM answers, we double check: does the article it cited actually contain the claim it made? This catches the LLM lying about its own sources. |
| **Eval set** | A list of test questions with known-good answers, used to score how well Sakan performs — like unit tests, but for AI answer quality. |
| **Faithfulness** | A score (0–1) measuring whether the answer's claims are actually backed up by the retrieved text, rather than invented. |
| **Streamlit** | A Python tool for building a simple web page/UI without needing to know HTML/CSS/JavaScript. We'll use it for the chat interface. |

### The big picture pipeline
```
Law PDFs  →  chunked into Articles  →  embedded + keyword-indexed
   →  user asks a question  →  hybrid search finds relevant Articles
   →  LLM answers ONLY from those Articles, citing them
   →  we verify the citation is real  →  answer shown in a web page
```

---

## Part 2 — What you need installed before we begin

We will handle this together as **Step 0**, but for awareness, you'll need:
- **Python** (3.10 or newer)
- A **code editor** (you're already set up with Claude Code, so we're good)
- A **free or paid API key** from OpenAI or Anthropic (for the LLM step — not
  needed until Day 3)
- A **GitHub account** (for deploying the demo later, Day 5–6)
- Internet access to download the law PDFs

You do **not** need to know these tools yet — I'll explain each one the moment
we use it.

---

## Part 3 — The Plan, Step by Step

Each step below is small on purpose. We will do ONE step, you'll see it work,
then we move to the next. Steps are grouped into the 6 "days" from the build
guide, but each day is broken into smaller sub-steps so nothing feels like a
big leap.

### Day 1 — Get the real law text, and write test questions FIRST

> **Why write test questions before building anything?** This feels backwards,
> but it's the most important habit in this project. If we write the eval
> questions *after* the system exists, we'll unconsciously write questions the
> system already handles well. Writing them first means they're an honest test.

- [x] **Step 1.1 — Project skeleton.** Folder structure (`data/`, `eval/`,
  empty placeholder files `ingest.py`/`index.py`/`retrieve.py`/`generate.py`/
  `app.py`), a Python virtual environment (`venv/`), and a git repo were
  already set up before we started.
- [x] **Step 1.2 — Download source documents.** Downloaded the real PDFs
  directly from official Dubai Land Department URLs into `data/raw/`:
  `law_26_2007.pdf` (10 pages), `law_33_2008.pdf` (6 pages), and
  `tenancy_guide.pdf` (42 pages — this one turned out to also contain the full
  text of **Decree No. 43 of 2013**, the tiered rent-increase calculator, plus
  both laws reprinted as appendices, so we don't need a separate live-scrape
  step for the percentage table).
- [x] **Step 1.3 — Skim & sanity-check the PDFs.** Extracted text with
  `pdftotext` and confirmed all three are real selectable text (no OCR
  needed), and that `Article (N)` headings are formatted consistently — the
  chunking regex from the guide will work. Also discovered an important
  wrinkle for Day 2: Law No. 33 of 2008 explicitly **supersedes** Articles
  2, 3, 4, 9, 13, 14, 15, 25, 26, 29, and 36 of the 2007 law — so those
  article numbers exist in *two* versions across our two law PDFs, and only
  the newer one is current law. We'll need to tag chunks with which document
  they came from so retrieval/generation can prefer the current version.
- [x] **Step 1.4 — Write 36 eval questions** in `eval/questions.json` (18
  factual, 9 edge-case, 9 out-of-scope — including a couple designed to
  specifically trip up a system that doesn't know about the Law 33/2008
  supersession, and one prompt-injection attempt). Written and grounded
  against the actual downloaded text, before any retrieval/generation code
  exists.

**Checkpoint for Day 1:** You have real source PDFs saved locally, and a
question file that defines what "success" looks like — before a single line of
retrieval code exists.

---

### Day 2 — Turn the PDFs into a searchable index

- [x] **Step 2.1 — `ingest.py`: parse and clean.** Extracted raw text with
  `pypdf`. Found and fixed two real data-quality bugs along the way: (1) the
  source PDFs have a font defect that silently drops the letter "i" from
  ligatures like "fi"/"ffi" ("five" → "fve", "official" → "offcial") — fixed
  with a checked replacement table; (2) page headers/footers/footnote
  markers were landing mid-sentence wherever a PDF page break fell, not just
  at page edges — fixed with a line-level junk filter.
- [x] **Step 2.2 — Chunk by Article boundary.** Split on `Article (N)`
  **header lines specifically** (not just any regex match) — legal text is
  full of in-text cross-references like "...as stipulated in Article (9) of
  this Law," which a naive regex would misfire on and treat as new article
  boundaries. Also solved the Law 33/2008 numbering collision from Step
  1.3: chunks are tagged with both `law` (which instrument's numbering:
  `26/2007`, `33/2008`, or `Decree 43/2013`) and `article_no`, plus
  `is_current` — so "Article 9" from the outdated 2007 text and the current
  2008 replacement text never get confused with each other. Non-article
  prose (the EJARI registration guide) is chunked by section heading
  instead, with a 1500-character size cap as a safety net.
- [x] **Step 2.3 — Save processed chunks** to `data/processed/chunks.json` —
  78 chunks total (54 law/decree articles + 24 EJARI-guide prose sections).
  Spot-checked the trickiest ones by hand (Article 9's two versions, Article
  25's new "landlord wants to sell" ground, the Decree 43/2013 percentage
  table) against the actual PDF text to confirm they're correct.
- [x] **Step 2.4 — `index.py`: build the vector index.** Embeds every chunk
  with ChromaDB's built-in local embedding model (no API key needed yet —
  that's only for Day 3's answer-writing LLM) and stores it in `chroma_db/`
  (gitignored — it's a regenerable build artifact, not source).
- [x] **Step 2.5 — `index.py`: build the BM25 keyword index** over the same
  chunks with `rank_bm25`. Rebuilt in memory every run rather than saved to
  disk — cheap enough at 78 chunks that persisting it isn't worth the extra
  moving part.
- [x] **Step 2.6 — Quick manual test.** `index.py` runs both indexes against
  a real eval question (f01, about contract registration) as a smoke test.
  Found something worth carrying into Day 3: vector search's #1 result was
  the **outdated** original Article 4, not the current amended one —
  exactly the `is_current` mix-up we built the metadata to catch. Day 3's
  retrieval logic needs to explicitly prefer `is_current: true` chunks, not
  just take the top-ranked match as-is.

**Checkpoint for Day 2:** Given any question, we can retrieve relevant law
chunks two different ways (meaning-based and keyword-based) — even though we
don't have an AI-generated answer yet.

---

### Day 3 — Make it answer questions, grounded in the text

- [ ] **Step 3.1 — `retrieve.py`: hybrid search.** Combine vector + BM25
  results using Reciprocal Rank Fusion into one ranked list of top chunks.
- [ ] **Step 3.2 — Get an LLM API key working.** Small end-to-end "hello
  world" call to OpenAI or Anthropic to confirm your key and billing are set
  up, before wiring it into the real pipeline.
- [ ] **Step 3.3 — `generate.py`: the guarded prompt.** Write the system
  prompt that forces the model to answer *only* from retrieved context, cite
  `(Article N)` per claim, and reply `"Not covered by this source set."` when
  it can't find an answer.
- [ ] **Step 3.4 — Wire it together.** question → retrieve → generate →
  printed answer, runnable from the command line.
- [ ] **Step 3.5 — Manual smoke test.** Ask it 3–4 real questions and 1
  deliberately out-of-scope question, and read the answers yourself.

**Checkpoint for Day 3:** You can ask Sakan a question in your terminal and get
a cited answer, or an honest refusal.

---

### Day 4 — Prove it's trustworthy (this is the resume-worthy part)

- [ ] **Step 4.1 — Install & configure `deepeval`.**
- [ ] **Step 4.2 — `eval/test_faithfulness.py`.** Run every question from
  `eval/questions.json` through Sakan, score each with the `FaithfulnessMetric`
  (do the claims in the answer actually appear in the retrieved text?).
- [ ] **Step 4.3 — Citation-verification check.** After generation, confirm
  the article number cited actually contains the claim made — a second,
  independent safety net beyond the faithfulness score.
- [ ] **Step 4.4 — Score refusal accuracy.** Specifically check: for every
  out-of-scope question, did Sakan correctly refuse instead of guessing?
- [ ] **Step 4.5 — Tune and re-run.** If any question fails, adjust the
  prompt, chunking, or retrieval `k` value, then re-run the eval to see the
  score improve. This before/after is exactly what goes in the README later.

**Checkpoint for Day 4:** A numeric, repeatable score for "does this system
tell the truth" — not just your gut feeling from Day 3's manual test.

---

### Day 5 — Give it a face, and put it online

- [ ] **Step 5.1 — `app.py`: Streamlit chat UI.** A text box for the question,
  an answer area, and the cited article shown inline/expandable underneath.
- [ ] **Step 5.2 — Add the required disclaimer** ("informational demo, not
  legal advice — verify current figures with DLD") visibly in the UI.
- [ ] **Step 5.3 — Local run-through.** Launch it on your machine, click
  through the golden path and a couple of edge cases yourself.
- [ ] **Step 5.4 — Push to GitHub.**
- [ ] **Step 5.5 — Deploy on Streamlit Community Cloud** (free, connects
  directly to your GitHub repo) and get a public link.

**Checkpoint for Day 5:** A live URL you can send anyone, that answers a
real question in under ~10 seconds.

---

### Day 6 — Make it a portfolio piece

- [ ] **Step 6.1 — README.md.** Explain the problem, the architecture (with a
  simple diagram), the tech choices and why, and how to run it locally.
- [ ] **Step 6.2 — Publish the eval numbers** (faithfulness score, refusal
  accuracy) in the README, including the before/after from Day 4's tuning.
- [ ] **Step 6.3 — Record a 60-second demo clip** showing a real question
  answered with citation, and one refusal.
- [ ] **Step 6.4 — Final polish pass.** Broken links, typos, stray debug
  code, secrets accidentally committed (API keys!) — check for all of these.

**Checkpoint for Day 6 — Definition of Done:**
- [ ] Live demo link, answers in under ~10 seconds
- [ ] Every answer shows a source citation, or an explicit refusal
- [ ] Eval numbers stated in the README, with a before/after
- [ ] Disclaimer visible in the UI

---

## How we'll actually work through this

Since you're building step by step: tell me when you're ready and we'll start
at **Step 0 (setup)** then **Step 1.1**. I'll do one step at a time, explain
what the code/command does as we go, show you the result, and only move on
once you've confirmed it makes sense and works. If a term comes up that isn't
in the glossary above, just ask and I'll explain it in place.
