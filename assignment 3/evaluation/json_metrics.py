"""
evaluation/json_metrics.py
===========================
Computes automatic metrics for the JSON evaluation set at each checkpoint:

  - JSON validity rate
  - Schema compliance rate
  - Exact-match accuracy
  - Field-level F1 (for extraction tasks)
  - Common error taxonomy (categorised parsing failures)
  - ROUGE-L and BERTScore for all tasks

Usage:
    python evaluation/json_metrics.py --config config.yaml --checkpoint 2
    python evaluation/json_metrics.py --config config.yaml --checkpoint all
"""

import argparse
import json
import re
import yaml
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_json(text: str) -> Optional[dict]:
    """Try to parse JSON from model output, stripping markdown fences."""
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def classify_json_error(text: str) -> str:
    """Classify the type of JSON formatting error."""
    text_stripped = text.strip()
    if not text_stripped:
        return "empty_output"
    if re.search(r"[\{|\[]", text_stripped) is None:
        return "no_json_structure"
    # Check for common issues
    if re.search(r"(?<!\")(\w+)(?!\"):", text_stripped):
        return "unquoted_keys"
    if re.search(r"[\{\[,]\s*[\}\]]", text_stripped):
        return "trailing_comma"
    if re.search(r"'[^']*'", text_stripped) and '"' not in text_stripped:
        return "single_quotes"
    if text_stripped.count("{") != text_stripped.count("}"):
        return "mismatched_braces"
    if text_stripped.count("[") != text_stripped.count("]"):
        return "mismatched_brackets"
    return "other_parse_error"


def check_schema_compliance(parsed: dict, reference: dict) -> bool:
    """
    Check if the parsed JSON has the same top-level keys as the reference.
    A lightweight proxy for schema compliance.
    """
    if not isinstance(parsed, dict) or not isinstance(reference, dict):
        return isinstance(parsed, type(reference))
    return set(parsed.keys()) == set(reference.keys())


def field_level_f1(predicted: dict, reference: dict) -> tuple[float, float, float]:
    """
    Compute field-level precision, recall, and F1 for extraction tasks.
    Checks key presence and string-normalised value match.
    """
    if not isinstance(predicted, dict) or not isinstance(reference, dict):
        return 0.0, 0.0, 0.0

    def normalise(v) -> str:
        return str(v).lower().strip()

    ref_pairs  = {k: normalise(v) for k, v in reference.items()}
    pred_pairs = {k: normalise(v) for k, v in predicted.items()}

    if not ref_pairs:
        return 1.0, 1.0, 1.0

    tp = sum(
        1 for k, v in ref_pairs.items()
        if k in pred_pairs and pred_pairs[k] == v
    )
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall    = tp / len(ref_pairs)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Main metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(responses: list, checkpoint_id: int) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Aggregators
    valid_json        = 0
    schema_compliant  = 0
    exact_matches     = 0
    error_types       = Counter()
    field_precisions  = []
    field_recalls     = []
    field_f1s         = []
    rouge1_scores     = []
    rouge2_scores     = []
    rougeL_scores     = []
    task_breakdown    = defaultdict(lambda: {"total": 0, "valid": 0, "exact": 0})
    hyps, refs        = [], []

    total = len(responses)

    for ex in responses:
        response  = ex.get("response", "")
        reference = ex.get("reference", "")
        task_type = ex.get("task_type", "unknown")

        task_breakdown[task_type]["total"] += 1

        # --- JSON validity ---
        parsed_response  = extract_json(response)
        parsed_reference = extract_json(reference) if reference else None

        if parsed_response is not None:
            valid_json += 1
            task_breakdown[task_type]["valid"] += 1

            # Schema compliance
            if parsed_reference is not None and check_schema_compliance(parsed_response, parsed_reference):
                schema_compliant += 1

            # Exact match (normalised JSON)
            if parsed_reference is not None:
                try:
                    if json.dumps(parsed_response, sort_keys=True) == json.dumps(parsed_reference, sort_keys=True):
                        exact_matches += 1
                        task_breakdown[task_type]["exact"] += 1
                except Exception:
                    pass

            # Field-level F1 for extraction tasks
            if task_type == "json_extraction" and parsed_reference is not None:
                p, r, f = field_level_f1(parsed_response, parsed_reference)
                field_precisions.append(p)
                field_recalls.append(r)
                field_f1s.append(f)

        else:
            error_types[classify_json_error(response)] += 1

        # ROUGE (text-level)
        if reference:
            rouge = scorer.score(reference, response)
            rouge1_scores.append(rouge["rouge1"].fmeasure)
            rouge2_scores.append(rouge["rouge2"].fmeasure)
            rougeL_scores.append(rouge["rougeL"].fmeasure)
            hyps.append(response)
            refs.append(reference)

    # BERTScore
    bert_p = bert_r = bert_f1 = None
    if hyps:
        try:
            P, R, F = bert_score_fn(hyps, refs, lang="en", verbose=False)
            bert_p  = float(P.mean())
            bert_r  = float(R.mean())
            bert_f1 = float(F.mean())
        except Exception as e:
            print(f"  [warn] BERTScore failed: {e}")

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "checkpoint":            checkpoint_id,
        "total_examples":        total,
        "json_validity_rate":    round(valid_json / total, 4) if total else 0,
        "schema_compliance_rate": round(schema_compliant / valid_json, 4) if valid_json else 0,
        "exact_match_rate":      round(exact_matches / total, 4) if total else 0,
        "field_level_precision": avg(field_precisions),
        "field_level_recall":    avg(field_recalls),
        "field_level_f1":        avg(field_f1s),
        "rouge1":                avg(rouge1_scores),
        "rouge2":                avg(rouge2_scores),
        "rougeL":                avg(rougeL_scores),
        "bertscore_precision":   round(bert_p, 4) if bert_p else None,
        "bertscore_recall":      round(bert_r, 4) if bert_r else None,
        "bertscore_f1":          round(bert_f1, 4) if bert_f1 else None,
        "error_taxonomy":        dict(error_types.most_common()),
        "task_breakdown":        {
            t: {
                "total":        v["total"],
                "validity_rate": round(v["valid"] / v["total"], 4) if v["total"] else 0,
                "exact_rate":   round(v["exact"] / v["total"], 4) if v["total"] else 0,
            }
            for t, v in task_breakdown.items()
        },
    }


# ---------------------------------------------------------------------------
# Alpaca automatic metrics (ROUGE, BERTScore)
# ---------------------------------------------------------------------------

def compute_alpaca_metrics(responses: list, checkpoint_id: int) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    hyps, refs = [], []
    lengths    = []

    for ex in responses:
        response  = ex.get("response", "")
        reference = ex.get("reference", "")
        lengths.append(len(response.split()))

        if reference:
            rouge = scorer.score(reference, response)
            rouge1_scores.append(rouge["rouge1"].fmeasure)
            rouge2_scores.append(rouge["rouge2"].fmeasure)
            rougeL_scores.append(rouge["rougeL"].fmeasure)
            hyps.append(response)
            refs.append(reference)

    bert_f1 = None
    if hyps:
        try:
            _, _, F = bert_score_fn(hyps, refs, lang="en", verbose=False)
            bert_f1 = float(F.mean())
        except Exception as e:
            print(f"  [warn] BERTScore failed: {e}")

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    return {
        "checkpoint":         checkpoint_id,
        "total_examples":     len(responses),
        "rouge1":             avg(rouge1_scores),
        "rouge2":             avg(rouge2_scores),
        "rougeL":             avg(rougeL_scores),
        "bertscore_f1":       round(bert_f1, 4) if bert_f1 else None,
        "avg_response_length": round(sum(lengths) / len(lengths), 1) if lengths else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_checkpoint(ckpt: int, cfg: dict):
    paths_cfg = cfg["paths"]
    out_dir   = Path("logs")
    out_dir.mkdir(exist_ok=True)

    ckpt_dir = Path(paths_cfg[f"checkpoint{ckpt}_dir"])

    # --- JSON metrics ---
    json_path = ckpt_dir / "json_responses.jsonl"
    if json_path.exists():
        json_responses = load_jsonl(str(json_path))
        json_metrics   = compute_metrics(json_responses, ckpt)
        out_path       = out_dir / f"json_metrics_ckpt{ckpt}.json"
        with open(out_path, "w") as f:
            json.dump(json_metrics, f, indent=2)
        print(f"\n[Checkpoint {ckpt}] JSON Metrics:")
        print(f"  Validity:        {json_metrics['json_validity_rate']:.1%}")
        print(f"  Schema Compliance: {json_metrics['schema_compliance_rate']:.1%}")
        print(f"  Exact Match:     {json_metrics['exact_match_rate']:.1%}")
        print(f"  Field-level F1:  {json_metrics['field_level_f1']}")
        print(f"  ROUGE-L:         {json_metrics['rougeL']}")
        print(f"  BERTScore F1:    {json_metrics['bertscore_f1']}")
        print(f"  → Saved to {out_path}")
    else:
        print(f"[warn] No JSON responses found for checkpoint {ckpt} at {json_path}")
        json_metrics = {}

    # --- Alpaca auto-metrics ---
    alpaca_path = ckpt_dir / "alpaca_responses.jsonl"
    if alpaca_path.exists():
        alpaca_responses = load_jsonl(str(alpaca_path))
        alpaca_metrics   = compute_alpaca_metrics(alpaca_responses, ckpt)
        out_path         = out_dir / f"alpaca_metrics_ckpt{ckpt}.json"
        with open(out_path, "w") as f:
            json.dump(alpaca_metrics, f, indent=2)
        print(f"\n[Checkpoint {ckpt}] Alpaca Auto-Metrics:")
        print(f"  ROUGE-1:    {alpaca_metrics['rouge1']}")
        print(f"  ROUGE-L:    {alpaca_metrics['rougeL']}")
        print(f"  BERTScore F1: {alpaca_metrics['bertscore_f1']}")
        print(f"  Avg Length: {alpaca_metrics['avg_response_length']} tokens")
    else:
        print(f"[warn] No Alpaca responses for checkpoint {ckpt}")
        alpaca_metrics = {}

    return json_metrics, alpaca_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default="all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoints = (
        [0, 1, 2] if args.checkpoint == "all"
        else [int(args.checkpoint)]
    )

    for ckpt in checkpoints:
        run_checkpoint(ckpt, cfg)

    print("\nAll JSON metrics complete.")


if __name__ == "__main__":
    main()
