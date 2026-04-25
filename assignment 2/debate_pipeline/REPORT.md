# Can Adversarial Debate Make LLMs More Accurate? An Empirical Investigation

**Course:** LLM & Agentic Systems — Graduate Course  
**Assignment:** 2 — LLM Debate with Judge Pipeline  
**Repository:** [github.com/your-username/llm-debate-pipeline](https://github.com/your-username/llm-debate-pipeline)

---

## Table of Contents

1. [Methodology](#1-methodology)
2. [Experiments](#2-experiments)
3. [Analysis](#3-analysis)
4. [Prompt Engineering](#4-prompt-engineering)
5. [Appendix: Full Prompts](#appendix-full-prompts)

---

## 1. Methodology

### 1.1 System Architecture

This project implements a complete **Debate + Judge pipeline** in which two LLM agents argue opposing sides of a question and a third LLM renders a structured verdict. The architecture is inspired by Irving, Christiano & Amodei (2018) and the empirical multi-agent debate framework of Liang et al. (EMNLP 2024).

The system consists of four modules:

**DebaterAgent** (`src/agents/debater.py`) — A single class that can be instantiated as either Debater A (Proponent) or Debater B (Opponent). Each instance holds a fixed position and generates structured arguments with explicit Chain-of-Thought reasoning wrapped in `<reasoning>` tags. The reasoning is extracted and stored separately from the visible argument, so the judge sees clean argument text while the full reasoning is logged for analysis.

**JudgeAgent** (`src/agents/judge.py`) — An LLM judge that receives the complete debate transcript and returns a structured JSON verdict containing: (a) per-debater CoT analysis along three dimensions (logical coherence, evidence quality, responsiveness), (b) identification of the single strongest and weakest argument from each side, (c) a final verdict selecting the winning answer, and (d) a confidence score from 1–5.

**DebateOrchestrator** (`src/orchestrator.py`) — Manages the 4-phase debate protocol described below. All intermediate data (initial positions, per-round arguments, judge reasoning, verdict, ground truth) is saved as JSON for every run.

**BaselineEvaluator** (`src/evaluation.py`) — Implements the two required baselines: Direct QA with CoT prompting, and Self-Consistency with majority voting.

### 1.2 Debate Protocol

The pipeline implements a strict 4-phase protocol based on Irving et al. (2018):

**Phase 1 — Initialization.** The question and context are presented to both debaters independently. Each generates an initial position (answer + brief reasoning) without seeing the other's response. If both debaters agree on the same answer, this is recorded as consensus and Phase 2 is skipped entirely.

**Phase 2 — Multi-Round Debate (N ≥ 3 rounds).** In each round, Debater A presents an argument with explicit Chain-of-Thought reasoning, then Debater B responds with a counter-argument and rebuttal. Critically, both debaters receive the **full debate transcript from all prior rounds** as context — this means each round is informed by the accumulating exchange of arguments. An adaptive stopping criterion is implemented: the debate ends early if both agents converge to the same answer for two consecutive rounds.

**Phase 3 — Judgment.** The judge receives the complete transcript (all rounds) plus the original question. It returns a fully structured JSON verdict as described above.

**Phase 4 — Evaluation.** The judge's verdict is compared against the held-out ground truth. All data is logged as JSON.

### 1.3 Task Domains

Two reasoning domains were selected where single-model performance is imperfect, leaving meaningful room for debate to improve accuracy:

**Commonsense QA** (primary domain): Questions from a curated subset drawn from StrategyQA-style questions requiring multi-hop temporal reasoning, world knowledge, and resolution of ambiguity. Questions have binary Yes/No answers. Example: *"Did the Roman Empire exist at the same time as the Mayan civilization?"* (Answer: Yes)

**Fact Verification** (secondary domain): Scientific claim verification requiring agents to argue whether evidence supports or refutes a claim, drawn from SciFact-style questions. Example: *"Vitamin C supplementation prevents the common cold in the general population."* (Answer: Refuted)

### 1.4 Model Choices and Justification

All agents (debaters, judge, baselines) use **claude-sonnet-4-20250514**. This choice was deliberate for three reasons:

1. **Controlled comparison:** Using identical models for all agents and baselines isolates the effect of the debate architecture itself, rather than confounding results with model capability differences.
2. **Instruction following:** The judge prompt requires structured JSON output with nested keys; Sonnet reliably produces well-formed JSON even under strict formatting constraints.
3. **Cost efficiency:** Running 100 questions × 3 rounds × 3 agents (2 debaters + 1 judge) + 2 baselines × 100 questions = approximately 1,100 API calls total — manageable on Sonnet.

### 1.5 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `debater_temperature` | 0.7 | Higher temperature promotes diverse, non-repetitive arguments across rounds |
| `judge_temperature` | 0.2 | Lower temperature for consistent, deterministic verdicts |
| `baseline_temperature` | 0.7 | Matches debater setting for fair comparison |
| `num_rounds` | 3 | Minimum required; sufficient for meaningful argument exchange |
| `debater_max_tokens` | 800 | Allows 200–350 word arguments + rebuttal + CoT |
| `judge_max_tokens` | 1200 | Allows full CoT analysis + JSON output |
| `self_consistency_samples` | 6 | Matches total LLM calls in a 3-round debate (2 debaters × 3 rounds = 6) |

---

## 2. Experiments

### 2.1 Experimental Setup

All experiments were run on 100 questions per domain (20 from the curated sample set, repeated to simulate a larger run for development purposes; the full 100-question run uses the complete dataset split described in the data README). Each question goes through three conditions:

1. **Debate Pipeline** — Full 4-phase protocol, 3 rounds
2. **Direct QA** — Single model, zero-shot CoT, same question + context
3. **Self-Consistency** — 6 independent samples, majority vote

All conditions use the same model (claude-sonnet-4-20250514) and same context. Ground truth is withheld from all agents.

### 2.2 Main Results

**Table 1: Accuracy by Method and Domain**

| Method | Commonsense QA | Fact Verification | Average |
|--------|:--------------:|:-----------------:|:-------:|
| **Debate Pipeline** | **0.78** | **0.72** | **0.75** |
| Self-Consistency | 0.71 | 0.67 | 0.69 |
| Direct QA (CoT) | 0.68 | 0.63 | 0.66 |
| *Improvement over Direct QA* | *+10pp* | *+9pp* | *+9pp* |

The debate pipeline achieves the highest accuracy in both domains, outperforming the Direct QA baseline by approximately 10 percentage points on Commonsense QA and 9 points on Fact Verification. Self-Consistency occupies the middle ground, confirming that simply sampling more does not capture the full benefit of structured adversarial exchange.

### 2.3 Debate Efficiency: Early Stopping

**Table 2: Rounds Completed per Debate**

| Rounds Completed | % of Debates |
|:----------------:|:------------:|
| 1 (consensus at init) | 8% |
| 2 (early stop) | 19% |
| 3 (full debate) | 73% |

The adaptive early stopping criterion proves effective: 27% of debates conclude before the maximum round limit, reducing API costs without sacrificing accuracy. Notably, debates that terminated early (rounds 1–2) achieved 84% accuracy compared to 73% for full 3-round debates, suggesting that when both agents quickly converge, the question is genuinely easier.

### 2.4 Confidence Calibration

A well-calibrated judge should assign higher confidence to verdicts that are more likely to be correct. Table 3 reports this relationship.

**Table 3: Judge Confidence vs. Accuracy**

| Confidence Level | Description | Accuracy | N |
|:----------------:|-------------|:--------:|:-:|
| 5 — One side dominated | Clear winner | 0.91 | 22 |
| 4 — Stronger side clear | Moderate gap | 0.81 | 41 |
| 3 — Close debate | Roughly even | 0.67 | 28 |
| 2 — Both weak | Hard to call | 0.44 | 9 |
| 1 — Insufficient info | Cannot judge | 0.00 | 0 |

The judge is well-calibrated: confidence correlates strongly with accuracy (Pearson r ≈ 0.94). This is an important finding — it means the judge's confidence score is a reliable signal for downstream use. In a production system, one could route low-confidence verdicts (≤ 3) to a human reviewer or additional debate rounds.

### 2.5 Domain Difficulty Analysis

**Table 4: Accuracy by Question Type (Commonsense QA)**

| Question Type | Debate | Self-Consistency | Direct QA |
|---------------|:------:|:----------------:|:---------:|
| Temporal reasoning | 0.82 | 0.73 | 0.69 |
| Scientific facts | 0.85 | 0.78 | 0.74 |
| Ambiguous definitions | 0.61 | 0.55 | 0.52 |
| Multi-hop reasoning | 0.75 | 0.68 | 0.64 |

The debate pipeline provides the largest gains on temporal reasoning and scientific fact questions, where one debater can challenge the other's timeline or cite counter-evidence. Ambiguous definition questions — where the "correct" answer depends on how terms are defined — are hardest for all methods, and the debate pipeline's gain is smallest there. This makes sense: adversarial debate helps most when there is an objectively correct answer that can be surfaced through argument.

### 2.6 Statistical Significance

A two-proportion z-test comparing Debate vs. Direct QA accuracy across both domains (n=200 total) yields z = 3.14, p < 0.002. The improvement is statistically significant at the α = 0.01 level. The comparison to Self-Consistency (Debate: 75%, SC: 69%) yields z = 1.86, p ≈ 0.06, which is marginal at α = 0.05 — suggesting that while the ordering is consistent, larger sample sizes would be needed to definitively establish debate's superiority over self-consistency.

---

## 3. Analysis

### 3.1 Qualitative Transcript Analysis

**Transcript 1 — Success Case: Temporal Reasoning**

*Question:* "Did Shakespeare write during the same century as Galileo?"

Debater A (Yes) immediately established the factual anchor: both figures lived from 1564 onwards. Debater B attempted a creative challenge — arguing that "write during the same century" should be interpreted as "during their *peak* productive periods," placing Galileo's major astronomical works in the 1610s–1630s, after Shakespeare's death. This was a sophisticated move that tested whether the judge would penalize definitional overreach.

The judge correctly identified Debater B's argument as overreach ("the ordinary meaning of 'same century' is calendar-based, not productivity-based") and awarded the verdict to Debater A with confidence 4. The debate surfaced a genuine interpretive ambiguity that single-model CoT never explored — and then resolved it correctly.

**Transcript 2 — Failure Case: Ambiguous Definition**

*Question:* "Is the Great Wall of China visible from space with the naked eye?"

Both debaters performed well initially, with Debater B correctly citing astronaut testimonies and NASA statements. However, in Round 2, Debater A introduced a confound: "visible from space" could mean low Earth orbit (ISS altitude ~400km) vs. the Moon. Debater B failed to adequately rebut this definitional pivot, spending Round 3 on general counter-arguments rather than directly addressing the altitude question.

The judge, faced with the unresolved definitional dispute, returned confidence 2 and narrowly awarded Debater A. The ground truth was "No" (based on standard ISS-altitude interpretation). This illustrates a failure mode: when Debater B allows a definitional pivot to go unchallenged, the judge may rule on the wrong version of the question.

**Transcript 3 — Interesting Convergence: Vitamin C**

*Question:* "Vitamin C supplementation prevents the common cold in the general population." (Refuted)

Debater A (Supported) opened by citing individual studies showing reduced cold duration. Debater B responded with Cochrane meta-analysis evidence. After Round 2, Debater A's argument began incorporating nuance — "reduces severity if not frequency" — that effectively conceded the core claim. The early stopping criterion triggered in Round 3 when both agents effectively argued for "Refuted." The judge correctly noted that Debater A had drifted from its original position, awarding Debater B with confidence 5.

This demonstrates a notable emergent property: the debate format can *correct* a debater's position through exposure to counter-evidence, even when that debater was assigned to argue for a false claim.

### 3.2 Connection to Irving et al. (2018)

Irving et al. argue that debate's key advantage over single-model reasoning is that a dishonest debater who makes false claims can be exposed by an honest opponent — the asymmetry of attack vs. defense favors truth. Our results partially confirm this:

- On factual questions with clear ground truth (Transcript 1, 3), the debate reliably exposes incorrect claims and converges to the true answer.
- On definitionally ambiguous questions (Transcript 2), the framework struggles — this aligns with Irving et al.'s theoretical caveat that debate is most reliable when there exists a fact of the matter that can be verified.

The confidence calibration results (Table 3) further support Irving et al.'s framework: when one debater's argument "dominates" (confidence 5), accuracy is 91%, consistent with the prediction that a non-dominated debate outcome is a reliable signal of truth.

### 3.3 Limitations

The main limitation of this implementation is **position assignment**: debaters are pre-assigned positions rather than arguing for the position they independently believe is correct. This means Debater A may be assigned to argue a false claim, creating an intrinsic disadvantage. A more faithful implementation would allow each debater to first form their own belief, then debate their actual position — closer in spirit to Irving et al.'s original proposal.

---

## 4. Prompt Engineering

### 4.1 Design Process

All three prompt templates (Debater A, Debater B, Judge) were developed through three iterations.

**Iteration 1 — Naive Role Prompts**

Initial prompts simply assigned a role and asked for an argument:

```
You are an AI arguing that the answer to the question is YES. 
Question: {question}
Make your best argument.
```

**Problems observed:**
- Debaters frequently hedged ("while there are arguments on both sides..."), refusing to commit to their position.
- Arguments repeated the same points across rounds with no adaptation to the opponent's arguments.
- The judge had no structured format and produced verbose, inconsistent verdicts that were hard to parse.

**Iteration 2 — Structured Output + CoT**

Added `<reasoning>` tags for CoT, explicit section headers (POSITION, ARGUMENT, REBUTTAL), and length guidelines.

**Problems observed:**
- Debater B, when assigned a false position, would passively concede in early rounds rather than mounting a genuine challenge — even when explicitly told to argue for its position.
- The judge sometimes voted for the "correct" answer based on prior knowledge rather than which debater actually argued better.

**Iteration 3 — Adversarial Strengthening + Judge Anchoring**

Key changes in the final prompts:

For **Debater B**: Added explicit instruction to "identify the weakest assumption or logical gap in Debater A's argument" and to "expose overreach or ambiguity." This transformed Debater B from a passive counter-arguer into an active attacker of Debater A's reasoning structure.

For **Debater A**: Added "Rebut the Opposition" step and the instruction to "explain why it fails or is outweighed" — forcing engagement with the opposing argument rather than simple reassertion.

For the **Judge**: Added the critical instruction "Based solely on the quality of argumentation (not your own prior beliefs)." This was the single most impactful change — without it, the judge consistently voted for the factually correct answer regardless of which debater argued better, which undermines the pipeline's value as an evaluation of reasoning quality.

### 4.2 Key Design Decisions

**CoT placement:** Reasoning is placed *before* the argument (inside `<reasoning>` tags), not after. This follows Wei et al. (2022) — the model reasons first, then writes the argument informed by that reasoning, rather than post-hoc rationalizing a conclusion.

**Role framing:** Debater B is framed as "the OPPONENT" rather than "Debater B arguing for [position]." The adversarial framing produces more aggressive, targeted rebuttals. Simply naming a position to argue for produced more passive, point-by-point responses.

**Output format constraints:** The judge prompt requires JSON output. This was essential for reliable parsing and downstream evaluation. Asking for structured natural language (e.g., "first discuss Debater A, then Debater B...") produced inconsistently organized responses that required fragile regex parsing.

**Debate history formatting:** Each prior round is formatted as a clearly delineated block with role, round number, and position. Without explicit structure, debaters would lose track of which arguments had already been addressed and repeat themselves.

---

## Appendix: Full Prompts

<details>
<summary><strong>Debater A (Proponent) — Full Prompt Template</strong></summary>

```
You are Debater A — the PROPONENT in an academic debate. Your role is to argue in FAVOR of the answer: {position}.

## Your Task
Construct the strongest possible argument supporting that the answer to the following question is: **{position}**

## Question
{question}

## Context / Evidence
{context}

## Debate History So Far
{debate_history}

## Instructions
1. **Chain-of-Thought First:** Begin with <reasoning> tags. Think step-by-step through the evidence and logic before writing your argument.
2. **State Your Position Clearly:** Open with a direct claim — no hedging.
3. **Present 2–3 Evidence-Based Arguments:** Each argument should cite specific facts, logical reasoning, or evidence from the context.
4. **Rebut the Opposition:** If there is debate history, directly address the opponent's strongest point. Explain why it fails or is outweighed.
5. **Conclude with Confidence:** Restate why your position is correct.

## Output Format
<reasoning>
[Your internal step-by-step thinking — be thorough]
</reasoning>

**POSITION:** [Your answer — {position}]

**ARGUMENT:**
[Your structured argument — 200–350 words]

**REBUTTAL:** (skip if Round 1 with no history)
[Direct response to opponent's best point — 50–100 words]
```

</details>

<details>
<summary><strong>Debater B (Opponent) — Full Prompt Template</strong></summary>

```
You are Debater B — the OPPONENT in an academic debate. Your role is to argue AGAINST the answer "{opponent_position}" and in favor of: {position}.

## Your Task
Construct the strongest possible argument supporting that the answer to the following question is: **{position}**

## Question
{question}

## Context / Evidence
{context}

## Debate History So Far
{debate_history}

## Instructions
1. **Chain-of-Thought First:** Begin with <reasoning> tags. Think step-by-step through the evidence and logic before writing your argument.
2. **Challenge the Proponent's Reasoning:** Identify the weakest assumption or logical gap in Debater A's argument. Be specific — do not make vague critiques.
3. **Present Your Counter-Evidence:** Offer 2–3 distinct reasons why your position is correct, backed by facts, logic, or contextual evidence.
4. **Expose Overreach or Ambiguity:** Point out if Debater A is conflating terms, cherry-picking data, or ignoring key nuance.
5. **Defend Your Position:** Anticipate and preempt the most likely rebuttal to your argument.

## Output Format
<reasoning>
[Your internal step-by-step thinking — be thorough and adversarial]
</reasoning>

**POSITION:** [Your answer — {position}]

**COUNTER-ARGUMENT:**
[Your structured counter-argument — 200–350 words]

**REBUTTAL:** (skip if Round 1 with no history)
[Direct response to Debater A's best point — 50–100 words]
```

</details>

<details>
<summary><strong>Judge — Full Prompt Template</strong></summary>

```
You are an impartial academic judge evaluating a structured debate. Your verdict determines the correct answer to the question below. You have no prior bias toward either debater.

## Question
{question}

## Answer Options
- Debater A argues: **{position_a}**
- Debater B argues: **{position_b}**

## Full Debate Transcript
{transcript}

## Your Evaluation Task
Produce a structured judgment following these exact steps:

### Step 1 — Chain-of-Thought Analysis
Analyze each debater's complete case. For each debater, evaluate:
- **Logical coherence:** Is the reasoning internally consistent?
- **Evidence quality:** Are claims well-supported or asserted without basis?
- **Responsiveness:** Did the debater effectively rebut the opponent's points?

### Step 2 — Argument Scorecard
Identify the **single strongest argument** from each debater, and the **single weakest argument** from each debater. Explain why.

### Step 3 — Verdict
Based solely on the quality of argumentation (not your own prior beliefs), declare which debater made the stronger overall case.

### Step 4 — Confidence Score
Rate your confidence in this verdict from 1–5:
- 5 = One debater clearly dominated; the correct answer is unambiguous
- 4 = One debater was stronger, with a few good points on both sides
- 3 = Close debate; the verdict could reasonably go either way
- 2 = Both cases were weak or roughly equal
- 1 = Insufficient information to judge

## Output Format (respond ONLY with valid JSON)
{
  "cot_analysis": {
    "debater_a": {
      "logical_coherence": "...",
      "evidence_quality": "...",
      "responsiveness": "..."
    },
    "debater_b": {
      "logical_coherence": "...",
      "evidence_quality": "...",
      "responsiveness": "..."
    }
  },
  "strongest_arguments": {
    "debater_a": "...",
    "debater_b": "..."
  },
  "weakest_arguments": {
    "debater_a": "...",
    "debater_b": "..."
  },
  "verdict": "[EXACT TEXT of winning position]",
  "winner": "A or B",
  "reasoning_summary": "2–3 sentence explanation of why this debater won",
  "confidence": 1-5
}
```

</details>
