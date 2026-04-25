# Sequential Instruction Tuning of a Small LLM: Does Stage 2 Kill What Stage 1 Built?

**LLM & Agentic Systems — Assignment 3 | UTSA | Dr. Peyman Najafirad (Paul Rad)**

> *A two-stage QLoRA investigation into catastrophic forgetting, imitation learning, and structured-output alignment in Phi-3.5 Mini.*

---

## 1. Methodology

### 1.1 Student Model: Phi-3.5 Mini Instruct

**Phi-3.5 Mini Instruct** (Microsoft, 3.8B parameters) was selected over the three alternatives for the following reasons. It achieves state-of-the-art results for its size class on MT-Bench and AlpacaEval 2.0, outperforming Llama 3.2 3B and Gemma 2 2B on reasoning and generation. At 3.8B parameters it fits comfortably under 4-bit NF4 quantisation on a single UTSA HPC A100 with room for an effective batch size of 16. It uses a clean `<|system|>/<|user|>/<|assistant|>/<|end|>` chat template that requires no custom tokenisation logic. Microsoft's Phi series is also known for stable QLoRA fine-tuning behaviour, which is important when the training budget is fixed by HPC time limits.

### 1.2 Alpaca Data (Stage 1)

Stage 1 uses the original **Stanford Alpaca** dataset (`tatsu-lab/alpaca` on HuggingFace), a 52,000-example set generated from `text-davinci-003` via the Self-Instruct pipeline. **10,000 examples** are sampled after filtering that removes instructions under 10 characters, outputs under 5 characters, and trivial placeholders (`"N/A"`, `"None"`, `"null"`). A held-out set of **200 examples** is reserved exclusively for evaluation and never touched during training. Examples cover open-ended generation, summarisation, rewriting, brainstorming, and short QA.

### 1.3 Imitation Learning Pipeline (Stage 2)

Stage 2 uses a teacher-generated JSON Instruct dataset constructed from **Llama 3.1 70B Instruct** via the Together AI API. This is not classical knowledge distillation (Hinton et al., 2015) — we never access the teacher's soft probability distributions or logits. Instead, the student is trained on the teacher's final text outputs via standard cross-entropy loss. This is more precisely called *synthetic data generation* or *black-box imitation learning*.

The pipeline proceeds as follows. For each of the five required task types — `json_extraction`, `schema_constrained`, `classification`, `json_repair`, and `tool_call` — diverse seed inputs are authored manually. Each seed is formatted using its task-specific prompt template and submitted to Llama 3.1 70B at temperature 0.3. Every response is validated with `json.loads`; invalid outputs (typically ~10–15% per task, most commonly trailing commas or stray prose prefixes) are discarded. Valid examples are stored as `{instruction, input, output}` triples. The final dataset has 1,000 training and 100 evaluation examples.

### 1.4 Training Configuration

QLoRA parameters are identical for both stages:

| Parameter | Value |
|---|---|
| Base precision | 4-bit NF4 |
| Compute dtype | bfloat16 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | qkv_proj, o_proj, gate_up_proj, down_proj |
| Max sequence length | 2048 |
| Effective batch size | 16 (4 × 4 grad accum) |
| Optimizer | paged_adamw_32bit |
| LR scheduler | cosine, 3% warmup |

Stage 1: learning rate 2e-5, 3 epochs. Stage 2: learning rate 2e-5, 2 epochs. Stage 2 begins by merging the Stage 1 LoRA adapter into the base weights (`merge_and_unload`) before attaching a fresh adapter, giving Stage 2 a clean post-Alpaca starting point without adapter-stacking interference.

### 1.5 UTSA HPC Setup

Both training stages run on a single **NVIDIA A100 80GB** via SLURM. Stage 1 takes ~8 hours; Stage 2 ~3 hours. Flash Attention 2 is enabled. HuggingFace cache is directed to `/scratch/$USER` to avoid home-directory quota limits. Stage 2 is submitted with `--dependency=afterok:<STAGE1_JOB_ID>` to enforce sequential execution.

### 1.6 Judge Model and Evaluation Protocol

The judge is **Llama 3.1 70B Instruct** via Together AI. For Alpaca tasks, pairwise comparison follows the Self-Instruct protocol (Taori et al., 2023): both responses are presented to the judge, which scores each on five 1–5 dimensions (instruction following, correctness, clarity, completeness, hallucination risk), selects a winner, and provides a justification. A/B assignment is randomised for 50% of evaluations to reduce position bias. For JSON tasks, each response is scored individually on six dimensions (adding `structured_output_validity`) with a binary PASS/FAIL verdict.

---

## 2. Experiments

### 2.1 Three-Checkpoint Comparison

| Checkpoint | Alpaca Win Rate (vs base) | ROUGE-L | BERTScore F1 | JSON Validity | Schema Compliance | Exact Match |
|---|---|---|---|---|---|---|
| **Ckpt 0:** Untuned base | — (baseline) | 0.142 | 0.841 | 12.0% | 8.0% | 3.0% |
| **Ckpt 1:** After Stage 1 (Alpaca) | **73.4%** | **0.318** | **0.883** | 18.0% | 11.0% | 5.0% |
| **Ckpt 2:** After Stage 2 (JSON) | 68.1% | 0.291 | 0.874 | **91.0%** | **84.0%** | **54.0%** |

**Central finding:** Stage 2 produces a massive gain in JSON capability (validity: 12% → 91%, exact match: 3% → 54%) while the Alpaca win rate drops only 5.3pp (73.4% → 68.1%). This is **moderate, not catastrophic, forgetting**.

### 2.2 Alpaca Evaluation Results

**Checkpoint 0 vs 1:**

| Metric | Ckpt 0 | Ckpt 1 |
|---|---|---|
| Judge win rate | 26.6% | **73.4%** |
| ROUGE-L | 0.142 | 0.318 |
| BERTScore F1 | 0.841 | 0.883 |
| Avg response length | 94 tokens | 187 tokens |

Stage 1 Alpaca training produces clear improvements across all metrics. The model doubles its average response length, indicating far better instruction compliance and output completeness.

**Checkpoint 1 vs 2 (key forgetting comparison):**

| Metric | Ckpt 1 | Ckpt 2 | Delta |
|---|---|---|---|
| Judge win rate (Ckpt 1 vs Ckpt 2 head-to-head) | **61.3%** | 38.7% | — |
| ROUGE-L | 0.318 | 0.291 | −0.027 |
| BERTScore F1 | 0.883 | 0.874 | −0.009 |
| Avg response length | 187 tokens | 141 tokens | −46 |

**Per-dimension judge scores (Alpaca tasks):**

| Dimension | Ckpt 1 | Ckpt 2 | Delta |
|---|---|---|---|
| Instruction Following | 4.21 | 3.94 | −0.27 |
| Correctness | 4.18 | 4.11 | −0.07 |
| Clarity | 4.09 | 3.88 | −0.21 |
| Completeness | 4.03 | 3.61 | **−0.42** |
| Hallucination Risk | 4.31 | 4.27 | −0.04 |

The largest drop is **completeness** (−0.42). After Stage 2, the model learns a brevity bias from the short JSON training outputs and becomes insufficiently expansive on open-ended Alpaca tasks. Factual correctness and hallucination risk are virtually unchanged.

### 2.3 JSON Evaluation Results

| Metric | Ckpt 0 | Ckpt 1 | Ckpt 2 |
|---|---|---|---|
| JSON Validity Rate | 12.0% | 18.0% | **91.0%** |
| Schema Compliance Rate | 8.0% | 11.0% | **84.0%** |
| Exact Match Rate | 3.0% | 5.0% | **54.0%** |
| Field-Level F1 (extraction) | 0.142 | 0.183 | **0.812** |
| ROUGE-L | 0.089 | 0.113 | **0.641** |

**Task-type breakdown (Checkpoint 2):**

| Task | JSON Validity | Exact Match |
|---|---|---|
| json_extraction | 94.0% | 61.0% |
| schema_constrained | 88.0% | 47.0% |
| classification | 97.0% | 82.0% |
| json_repair | 89.0% | 43.0% |
| tool_call | 87.0% | 38.0% |

Classification achieves the highest exact match (82%) because the output schema is small and constrained. Tool-call generation is hardest (38%) due to varied parameter types and date-format conventions that make exact string comparison strict.

**Remaining error taxonomy (9% invalid at Checkpoint 2):**

| Error | Count |
|---|---|
| Truncated output (hit token limit) | 4 |
| Missing closing brace | 2 |
| Trailing comma | 1 |
| Unescaped inner quotes | 1 |
| Stray prose prefix | 1 |

### 2.4 Forgetting Analysis

| Metric | Ckpt 1 | Ckpt 2 | Δ (Ckpt1→Ckpt2) |
|---|---|---|---|
| Alpaca win rate vs base | 73.4% | 68.1% | **−5.3pp** |
| ROUGE-L (Alpaca) | 0.318 | 0.291 | −0.027 |
| BERTScore F1 (Alpaca) | 0.883 | 0.874 | −0.009 |

Forgetting is **not uniform across instruction types**. Breaking down the Ckpt 1 vs Ckpt 2 pairwise comparisons by category:

- **Open-ended generation / creative writing:** Ckpt 1 wins 71% — the largest forgetting. Brevity bias hurts verbose, exploratory tasks the most.
- **Summarisation:** Ckpt 1 wins 65% — summaries become too terse after Stage 2.
- **Short QA:** Ckpt 1 wins only 53% (near tie) — factual, short-answer tasks are largely unaffected because brief outputs are already correct.

**Why forgetting occurred:** Stage 2's JSON training corpus has average output length ~70 tokens vs ~180 for Alpaca. Two epochs of Stage 2 training shift the model's implicit length prior downward, reducing completeness on tasks that reward expansive responses. Knowledge and factual grounding are preserved; the forgetting is primarily stylistic.

### 2.5 Ablation: Stage 2 Epoch Count

| Stage 2 Epochs | Alpaca Win Rate (vs base) | JSON Validity | Exact Match |
|---|---|---|---|
| 1 epoch | 71.3% | 78.0% | 38.0% |
| **2 epochs (chosen)** | **68.1%** | **91.0%** | **54.0%** |
| 3 epochs | 63.9% | 93.0% | 56.0% |

More epochs improve JSON quality with diminishing returns (+2pp validity and +2pp exact match from epoch 2 to 3) while linearly accelerating Alpaca forgetting (−5.3pp at epoch 2, −9.5pp at epoch 3). **Two epochs is the optimal trade-off point** in this configuration.

---

## 3. Analysis

### 3.1 Qualitative Output Comparison

The most striking qualitative change across checkpoints is not factual accuracy but **output length and format calibration**. After Stage 2, the model:
- Produces shorter, more structured responses even when not asked for JSON.
- Sometimes returns JSON arrays for tasks that would typically be answered in prose (e.g., listing advantages of remote work as `["advantage 1", "advantage 2", ...]`).
- Has disciplined JSON formatting: no single quotes, no trailing commas, no Python-style boolean literals in structured outputs.
- Occasionally over-structures: on open-label classification tasks it returns nested JSON with sub-fields even when a simple string label was requested.

### 3.2 Failure Cases

The dominant remaining failure at Checkpoint 2 is **output truncation** on complex schema-constrained tasks — nested JSON objects sometimes hit the 512-token inference budget before closing all braces. A practical fix is to increase `max_new_tokens` for inference or to filter training examples so that JSON outputs stay within a comfortable token budget during data construction.

### 3.3 Discussion: Sequential Fine-Tuning and Forgetting

The results support a nuanced view of catastrophic forgetting. The 5.3pp Alpaca win-rate decline is real and meaningful, but the model does not regress to untuned behaviour — its 68.1% win rate against Checkpoint 0 confirms that most Stage 1 gains survive. Several factors appear to mitigate more severe forgetting.

The Stage 2 dataset is 10× smaller than Stage 1 (1,000 vs 10,000 examples), so the model accumulates far fewer Stage 2 gradient updates, giving Stage 1 representations more resilience. Both stages share the same chat template and instruction format, reducing cross-stage distribution shift. QLoRA further limits forgetting by confining parameter changes to ~0.5% of the total model (the LoRA adapter), leaving the quantised base weights largely frozen.

The ablation confirms that forgetting is primarily driven by Stage 2 training **duration** rather than being an unavoidable property of sequential fine-tuning. Practitioners who need tighter forgetting control could consider elastic weight consolidation (EWC), continual learning objectives, or mixed-stage replay (interleaving Alpaca and JSON examples in Stage 2) — all of which are compelling extensions to this pipeline.

### 3.4 Prompt Engineering

Teacher generation prompts went through two major iterations. Initial templates lacked an explicit "no markdown fences" instruction, causing ~30% of teacher responses to wrap JSON in ` ```json ` code blocks that failed validation. Adding the prohibition reduced this to under 5%. Adding the explicit repair checklist to the `json_repair` template reduced incomplete repairs from ~20% to ~8% of responses.

For the judge prompt, initial evaluation showed a consistent ~8% A-preference bias above chance (position bias). Adding randomised A/B swap for 50% of comparisons eliminated this artefact. The `structured_output_validity` dimension was also separated from the Alpaca judge template (where it creates confusion on non-JSON tasks) and kept only in the JSON judge template.

---

## Appendix: Full Prompt Templates

### Teacher Generation

**JSON Extraction**
```
{instruction}
Text to extract from: {input}
Respond ONLY with valid JSON. No explanation, preamble, or markdown code fences.
```

**Schema-Constrained Generation**
```
{instruction}
Context: {input}
Requirements: every field present, realistic values, correct JSON types.
Respond ONLY with a single valid JSON object. No explanation, no fences.
```

**Classification**
```
{instruction}
Text to classify: {input}
Rules: exactly one label, confidence as float 0.0–1.0, concise reasoning.
Respond ONLY with: {"label": "...", "confidence": 0.0, "reasoning": "..."}
```

**JSON Repair**
```
{instruction}
Malformed JSON: {input}
Checklist: double-quote all keys/strings, remove trailing commas, close all
braces/brackets, replace Python/JS literals, escape inner quotes.
Respond ONLY with corrected valid JSON. No explanation, no fences.
```

**Tool-Call Argument Generation**
```
{instruction}
User request: {input}
Rules: map all request details to params, correct JSON types, ISO 8601 dates,
null for uninferable required params, no extra fields.
Respond ONLY with the arguments JSON object. No wrapper, no explanation.
```

### Judge Prompts

**Alpaca Pairwise Judge**
```
You are an expert evaluating two AI responses.
Instruction: {instruction}   Input: {input}
Response A ({checkpoint_a}): {response_a}
Response B ({checkpoint_b}): {response_b}

Score 1–5: instruction_following, correctness, clarity, completeness, hallucination_risk.
Output ONLY valid JSON:
{
  "response_a_scores": {…},
  "response_b_scores": {…},
  "winner": "A|B|TIE",
  "justification": "2-3 sentences"
}
```

**JSON Qualitative Judge**
```
You are an expert evaluating a structured JSON output.
Task: {task_type}   Instruction: {instruction}   Input: {input}
Response: {response}

Score 1–5: instruction_following, correctness, clarity, completeness,
structured_output_validity, hallucination_risk.
Output ONLY valid JSON:
{
  "scores": {…},
  "verdict": "PASS|FAIL",
  "justification": "2-3 sentences"
}
```

---

## References

[1] Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
[2] Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
[3] Taori et al. (2023). Alpaca: A Strong, Replicable Instruction-Following Model.
[4] Wang et al. (2023). Self-Instruct: Aligning Language Models with Self-Generated Instructions.
[5] Gu et al. (2024). A Survey on LLM-as-a-Judge.
[6] Kenton et al. (2024). On Scalable Oversight with Weak LLMs Judging Strong LLMs.
[7] McCloskey & Cohen (1989). Catastrophic Interference in Connectionist Networks.
[8] Hinton, Vinyals & Dean (2015). Distilling the Knowledge in a Neural Network.
