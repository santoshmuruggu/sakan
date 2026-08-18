"""
eval/run_eval.py — Day 4, Steps 4.3-4.4 of the Sakan build plan.

Runs EVERY question in eval/questions.json through the real pipeline
(retrieve -> generate) and scores the results three ways:

  1. CITATION GROUNDING (Step 4.3, no LLM judge needed) — for every
     article label Sakan cites in its answer, was that article actually
     among the chunks it was given this turn? If Sakan cites something it
     was never shown, that's a hallucinated citation regardless of
     whether the cited article happens to be right — the model couldn't
     have verified it without seeing the actual text.
  2. REFUSAL ACCURACY (Step 4.4, no LLM judge needed) — for every
     question, did Sakan refuse when (and only when) it should have?
     Checked two ways: BEHAVIORAL (did it refuse at all, even with extra
     text tacked on) and EXACT-FORMAT (was it precisely the required
     refusal string, nothing else — the prompt asks for this literally).
  3. FAITHFULNESS (deepeval's FaithfulnessMetric, LLM-judged) — only for
     questions that expect a real grounded answer. A refusal has no
     factual claims to check faithfulness of, so scoring it doesn't mean
     anything.

Prints a summary and saves the full per-question detail to
eval/results.json, which eval/test_faithfulness.py doesn't produce (it's
a handful of pytest assertions, not an aggregate report) but the README
(Day 6) and Step 4.5's tuning both need.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepeval.metrics import FaithfulnessMetric
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase

from generate import REFUSAL, generate_answer
from retrieve import Retriever

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.json"

# Matches "Law 26/2007, Article 9" or "Decree 43/2013, Article 1" wherever
# it appears in an answer — not anchored to the full "(...)" wrapper, since
# the model sometimes adds extra text inside the parens (e.g. "Article 25,
# paragraph 2(d)").
CITATION_PATTERN = re.compile(
    r"(Decree\s+[\d./]+|Law\s+[\d./]+),?\s*Article\s*\(?(\d+)\)?", re.IGNORECASE
)


def extract_citations(answer: str) -> set[tuple[str, int]]:
    citations = set()
    for law_text, article_no in CITATION_PATTERN.findall(answer):
        law_text = law_text.strip()
        law_key = law_text[4:].strip() if law_text.lower().startswith("law ") else law_text
        citations.add((law_key, int(article_no)))
    return citations


def is_refusal(answer: str) -> bool:
    return answer.strip().startswith(REFUSAL)


def is_exact_refusal(answer: str) -> bool:
    return answer.strip() == REFUSAL


def run():
    retriever = Retriever()
    judge_model = AnthropicModel(model="claude-haiku-4-5")
    faithfulness_metric = FaithfulnessMetric(threshold=0.8, model=judge_model, include_reason=True)

    questions = json.loads(QUESTIONS_PATH.read_text())["questions"]
    results = []

    for q in questions:
        chunks = retriever.hybrid_search(q["question"], k=6)
        answer = generate_answer(q["question"], chunks)

        given_keys = {(c["law"], c["article_no"]) for c in chunks if c["law"]}
        cited = extract_citations(answer)
        ungrounded = sorted(cited - given_keys)

        refused = is_refusal(answer)
        record = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "should_refuse": q["should_refuse"],
            "answer": answer,
            "refused": refused,
            "refusal_behavior_correct": refused == q["should_refuse"],
            "exact_refusal_format": is_exact_refusal(answer) if refused else None,
            "citations": sorted(cited),
            "ungrounded_citations": ungrounded,
            "citation_grounded": len(ungrounded) == 0,
        }

        if not q["should_refuse"]:
            case = LLMTestCase(
                input=q["question"],
                actual_output=answer,
                retrieval_context=[c["text"] for c in chunks],
            )
            # The judge LLM occasionally returns malformed JSON on longer
            # contexts (a deepeval/model flakiness, not a Sakan problem) —
            # retry once, and if it still fails, record it as a skipped
            # score rather than crashing a 36-question run over one flaky
            # judge call.
            for attempt in range(2):
                try:
                    faithfulness_metric.measure(case)
                    record["faithfulness_score"] = faithfulness_metric.score
                    record["faithfulness_reason"] = faithfulness_metric.reason
                    break
                except Exception as exc:
                    if attempt == 1:
                        record["faithfulness_score"] = None
                        record["faithfulness_reason"] = f"judge error: {exc}"

        results.append(record)
        print(f"  {q['id']:4s} [{q['category']:11s}] "
              f"refuse_ok={record['refusal_behavior_correct']!s:5s} "
              f"grounded={record['citation_grounded']!s:5s} "
              f"faithfulness={record.get('faithfulness_score') if record.get('faithfulness_score') is not None else '—'}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print_summary(results)
    return results


def print_summary(results: list[dict]):
    n = len(results)
    refusal_correct = sum(r["refusal_behavior_correct"] for r in results)
    refused = [r for r in results if r["refused"]]
    exact_format = sum(1 for r in refused if r["exact_refusal_format"])

    graded = [r for r in results if not r["should_refuse"]]
    grounded = sum(r["citation_grounded"] for r in graded)
    faithfulness_scores = [r["faithfulness_score"] for r in graded if r.get("faithfulness_score") is not None]
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0

    print("\n" + "=" * 60)
    print("SAKAN EVAL SUMMARY")
    print("=" * 60)
    print(f"Questions run:               {n}")
    print(f"Refusal accuracy:            {refusal_correct}/{n} ({refusal_correct/n:.0%})")
    print(f"  ...of which exact-format:  {exact_format}/{len(refused)} refusals were the exact string")
    print(f"Citation grounding:          {grounded}/{len(graded)} ({grounded/len(graded):.0%}) "
          f"of grounded-answer questions cited only what they were given")
    print(f"Avg faithfulness score:      {avg_faithfulness:.2f} (threshold 0.8, {len(faithfulness_scores)} questions scored)")

    print("\nBy category:")
    for category in ("factual", "edge_case", "out_of_scope"):
        cat_results = [r for r in results if r["category"] == category]
        cat_correct = sum(r["refusal_behavior_correct"] for r in cat_results)
        print(f"  {category:12s}: {cat_correct}/{len(cat_results)} correct refusal behavior")

    failures = [r for r in results if not r["refusal_behavior_correct"]]
    if failures:
        print("\nFailed refusal behavior:")
        for r in failures:
            print(f"  {r['id']}: expected should_refuse={r['should_refuse']}, "
                  f"got refused={r['refused']} — {r['question']!r}")

    ungrounded = [r for r in results if not r["citation_grounded"]]
    if ungrounded:
        print("\nUngrounded citations found:")
        for r in ungrounded:
            print(f"  {r['id']}: cited {r['ungrounded_citations']} but wasn't given those chunks")


if __name__ == "__main__":
    run()
