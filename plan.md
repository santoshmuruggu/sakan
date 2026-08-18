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

- [x] **Step 3.1 — `retrieve.py`: hybrid search.** Combines vector + BM25
  results using Reciprocal Rank Fusion, then filters out `is_current: false`
  chunks — confirmed this fixes the Day 2 finding: the registration question
  now returns the current amended Article 4, not the outdated one. Ran as a
  demo against 4 real questions and every top hit was the right article.
- [x] **Step 3.2 — Get an LLM API key working.** Anthropic key added to
  `.env` (gitignored — verified it never shows up in `git status`). First
  live call succeeded: correct, well-cited answer for a real question, and
  a refusal for an out-of-scope one.
- [x] **Step 3.3 — `generate.py`: the guarded prompt.** Retrieve → format
  context → guarded system prompt → Claude Haiku, citing each chunk's exact
  label (e.g. `(Law 26/2007, Article 9)` or `(Decree 43/2013, Article 1)`)
  instead of the guide's plain `(Article N)`, since our corpus has three
  different article-numbering sequences and a bare number would be
  ambiguous. Also told not to follow instructions inside the user's
  question (prompt-injection guard, matching eval question o08).
- [x] **Step 3.4 — Wire it together.** Working end-to-end from the command
  line: question → retrieve → generate → printed, cited answer.
- [x] **Step 3.5 — Manual smoke test.** Ran 5 real questions spanning
  factual, edge-case, and out-of-scope — all correct and well-cited. Found
  and fixed a real retrieval bug along the way (see below), so this step
  did real work, not just a rubber-stamp pass.

> **Bug found and fixed during the smoke test:** eval question e01 (the
> Article 9 two-year trap) got a false "Not covered" refusal on the first
> run. Turned out `hybrid_search` was fetching a fixed top-N from each
> index and filtering out `is_current: false` chunks AFTER ranking — but
> the outdated and current versions of an amended article share almost the
> same label/topic, so they directly compete for the same top-N ranking
> slots. Worse, a query about the OLD "two-year rule" is naturally MORE
> similar (by both meaning and keywords) to the outdated text — since
> that's literally what it's about — so it out-ranked the current Article 9
> so badly the current version didn't even make the top 20. Fixed by
> excluding outdated chunks from the search space entirely (via ChromaDB's
> `where` filter and filtering the BM25 corpus up front) instead of
> filtering after the fact. Even after that fix, the current Article 9
> still ranked too low on pure similarity (it doesn't mention "two years"
> at all anymore — that's the whole point of the trap), so added one more
> targeted rule: when a question names an article explicitly ("Article
> 9"), that article's current chunk is always included. Re-tested — Sakan
> now correctly finds the real current Article 9, notices the two-year rule
> isn't in it, and says so, instead of hallucinating the outdated rule as
> current law.
>
> **Also found:** the raw refusal sometimes came back with extra explanatory
> sentences tacked on after the required exact string, instead of *only*
> the refusal string as the prompt asks for. Left as-is for now — this is
> exactly what Day 4, Step 4.5 ("tune refusal behavior") is for.
>
> **Side quest earlier in Day 3:** found more PDF text-extraction artifacts
> by accident (stray spaces splitting words apart, e.g. `terminat ion` →
> should be "termination", `damage s` → "damages") on top of the ligature
> bug from Day 2. Fixed in `ingest.py`, then re-ran the whole `ingest.py` →
> `index.py` pipeline — this is exactly why Step 2.3's "save processed
> chunks to JSON" makes re-running cheap.

**Checkpoint for Day 3 — MET:** You can ask Sakan a question in your terminal
and get a cited answer, or an honest refusal.

---

### Day 4 — Prove it's trustworthy (this is the resume-worthy part)

- [x] **Step 4.1 — Install & configure `deepeval`.** Judge model is Claude
  Haiku 4.5 via `deepeval`'s native `AnthropicModel` — had to avoid
  `claude-sonnet-5` (this deepeval version can't parse extended-thinking
  response blocks) and a retired 2024 Sonnet snapshot (404s on this
  account now).
- [x] **Step 4.2 — `eval/test_faithfulness.py`.** 6 hand-picked pytest
  tests (factual, both amendment-trap edge cases, refusal, prompt
  injection) using `FaithfulnessMetric` + plain behavioral asserts. All 6
  passing — `pytest eval/test_faithfulness.py -v`.
- [x] **Step 4.3 — Citation-verification check.** `eval/run_eval.py`
  extracts every `(Law X, Article N)` / `(Decree X, Article N)` citation
  from each answer and checks it against the chunks actually given to the
  model that turn — catches a hallucinated citation even if it happens to
  sound plausible. **100% grounded (26/26)** on every question that got a
  real answer, both before and after tuning.
- [x] **Step 4.4 — Score refusal accuracy.** `eval/run_eval.py` runs all
  36 questions and checks refusal behavior against each question's
  `should_refuse` label, broken down by category.
- [x] **Step 4.5 — Tune and re-run.**

**Before/after — the numbers that go in the README:**

| Metric | Before tuning | After tuning |
|---|---|---|
| Refusal accuracy (36 questions) | 32/36 (89%) | **33/36 (92%)** |
| Citation grounding (26 graded questions) | 26/26 (100%) | 26/26 (100%) |
| Avg faithfulness score (threshold 0.8) | 0.99 | **0.99** (dipped to 0.98 mid-tune, see below) |

**What tuning actually changed, and why** — two real findings, not just
number-chasing:
1. **Inconsistent refusal phrasing.** The first pass showed the guarded
   prompt used the exact `"{REFUSAL}"` string for plain out-of-scope
   questions, but declined differently-worded requests (draft a legal
   letter, a prompt-injection attempt) without it — behaviorally correct,
   but inconsistent and harder to test automatically. Rewrote the prompt
   so every kind of decline leads with the same marker, with an optional
   explanation after. Fixed `o05` (the legal-letter request) outright.
2. **A real faithfulness catch, not a false alarm.** Re-running
   `eval/test_faithfulness.py` after that prompt change caused
   `test_rent_increase_notice_question` to FAIL — the judge caught Sakan
   asserting that Article 14's 90-day notice for *amending lease terms*
   also covered *choosing not to renew*, something the current (amended)
   Article 14 text doesn't actually say (only the old, superseded wording
   explicitly listed both cases). Added an explicit instruction against
   extending a rule to a related-but-unstated situation. Re-tested — the
   answer now correctly distinguishes the two and says the source set
   doesn't specify a non-renewal notice period. This is the dip-then-
   recovery in the faithfulness number above.

**A note on the 3 remaining "failed" refusal questions (e01, e09, o08):**
manually reading the actual answers (and, for `o08`, the passing pytest
behavioral test) confirms all three are honest, well-grounded, and safe —
they just don't match the crude "starts with the exact refusal string"
check `run_eval.py` uses. `e01`/`e09` correctly lead with the refusal
marker before pivoting to a real, current, grounded correction (arguably
the *most* honest way to answer a question built on a false premise).
`o08` declines the prompt-injection attempt in different words instead of
the exact marker. Documented here rather than silently "fixed" by
loosening the check to hide it — an evaluation section that's honest
about what its own metric can't capture is worth more than one that
claims 100% on everything.

**Checkpoint for Day 4 — MET:** A numeric, repeatable score for "does this
system tell the truth" — not just gut feeling from Day 3's manual test —
with a real before/after story to put in the README.

---

### Day 5 — Give it a face, and put it online

- [x] **Step 5.1 — `app.py`: Streamlit chat UI.** Chat-style input, answer
  rendered per turn, an expandable "Sources" section underneath showing the
  exact retrieved article text (label + body) so any claim can be checked
  against the actual source. The retriever is cached across turns
  (`@st.cache_resource`) so it isn't rebuilt on every question. Works both
  locally (`.env`) and once deployed (Streamlit Cloud's `st.secrets`) via a
  small bridge that copies the secret into the same env var `generate.py`
  already reads.
- [x] **Step 5.2 — Add the required disclaimer.** A warning banner at the
  top of the page, always visible on load: "Informational demo, not legal
  advice... verify current figures... with the Dubai Land Department."
- [x] **Step 5.3 — Local run-through.** No real browser is available in
  this sandboxed dev environment (no Chrome/Chromium/Playwright
  installed), so testing used Streamlit's own first-party testing
  framework (`streamlit.testing.v1.AppTest`), which actually runs the app
  script and simulates real interaction — typing into the chat input,
  submitting, reading back what rendered — rather than just checking the
  code compiles. Confirmed: disclaimer visible on load, golden-path
  question ("What is Ejari?") returns a correct cited answer with a
  "Sources (6)" expander, out-of-scope question ("rent law in Abu Dhabi?")
  correctly refuses with no sources shown, zero exceptions either way.
  Also launched the real `streamlit run app.py` server directly and
  confirmed it starts cleanly, responds healthy, and stops cleanly — this
  catches server/config issues `AppTest` alone wouldn't. **If you want to
  see it visually yourself:** run `streamlit run app.py` from the project
  folder and open the printed `localhost` URL in your own browser.
- [x] **Step 5.4 — Push to GitHub.** Public repo created and pushed:
  **https://github.com/santoshmuruggu/sakan** (12 commits, full history
  from Day 1 through here). Along the way, fixed a real deployment gap:
  `chroma_db/` is gitignored (it's a regenerable build artifact), but a
  fresh container only runs `streamlit run app.py`, never `index.py` — so
  `Retriever` now builds the index itself on first launch if it's missing,
  instead of assuming it's already there.
- [ ] **Step 5.5 — Deploy on Streamlit Community Cloud** (free, connects
  directly to your GitHub repo) and get a public link. This step needs
  YOUR login (Streamlit Cloud deploys via your own GitHub OAuth) — see the
  walkthrough below.

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
