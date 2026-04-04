# LLM Debate Pipeline

> **Assignment 2 — LLM & Agentic Systems (Graduate Course)**  
> Building Adversarial Multi-Agent Reasoning Systems

A complete implementation of a Debate + Judge multi-agent pipeline in which two LLM agents argue opposing sides of a question, and a third LLM serves as judge — based on Irving, Christiano & Amodei (2018) and Liang et al. (EMNLP 2024).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 3. Run the web UI

```bash
streamlit run ui/app.py
```

Then open [http://localhost:8501](http://localhost:8501).

### 4. Run a single debate (CLI)

```bash
python scripts/run_debate.py \
  --question "Did Shakespeare live during the same century as Galileo?" \
  --answer "yes" \
  --position_a "yes" \
  --position_b "no" \
  --context "Shakespeare: 1564–1616. Galileo: 1564–1642."
```

### 5. Run all experiments

```bash
# Quick test (20 questions)
python scripts/run_experiments.py --domain commonsense_qa --n 20

# Full experiment (100 questions, both domains)
python scripts/run_experiments.py --all --n 100
```

### 6. Generate results tables

```bash
python scripts/evaluate.py --results_dir results/
```

---

## Project Structure

```
debate_pipeline/
├── config.yaml               ← All hyperparameters (nothing hardcoded)
├── requirements.txt
├── README.md
├── REPORT.md                 ← Blog post / write-up
│
├── src/
│   ├── agents/
│   │   ├── debater.py        ← DebaterAgent (Proponent & Opponent)
│   │   └── judge.py          ← JudgeAgent (verdict + CoT analysis)
│   ├── orchestrator.py       ← 4-phase debate protocol
│   └── evaluation.py         ← Baselines (Direct QA, Self-Consistency) + metrics
│
├── prompts/
│   ├── debater_a.txt         ← Proponent prompt template
│   ├── debater_b.txt         ← Opponent prompt template
│   └── judge.txt             ← Judge prompt template
│
├── data/
│   └── sample_questions.json ← 20 curated questions (commonsense QA + fact verification)
│
├── scripts/
│   ├── run_debate.py         ← Single debate (CLI)
│   ├── run_experiments.py    ← Full experiment suite
│   └── evaluate.py           ← Read logs → produce tables
│
├── ui/
│   └── app.py                ← Streamlit web UI
│
├── logs/                     ← Auto-generated debate transcripts (JSON)
└── results/                  ← Auto-generated summary metrics (JSON)
```

---

## Debate Protocol

The pipeline implements a 4-phase debate protocol:

**Phase 1 — Initialization**  
Both debaters independently generate their initial position without seeing the other's response. If they agree, debate is skipped.

**Phase 2 — Multi-Round Debate (N ≥ 3 rounds)**  
Debater A argues for its position, then Debater B responds. Both receive the full prior transcript as context. Early stopping triggers if both agents converge for 2 consecutive rounds.

**Phase 3 — Judgment**  
The judge receives the complete transcript and produces: (a) CoT analysis of both debaters, (b) strongest/weakest argument identification, (c) final verdict, (d) confidence score (1–5).

**Phase 4 — Evaluation**  
Judge verdict compared against ground truth. All data logged as JSON.

---

## Configuration

All hyperparameters live in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.debater` | `claude-sonnet-4-20250514` | Model for debaters |
| `model.judge` | `claude-sonnet-4-20250514` | Model for judge |
| `generation.debater_temperature` | `0.7` | Debater creativity |
| `generation.judge_temperature` | `0.2` | Judge consistency |
| `debate.num_rounds` | `3` | Number of debate rounds |
| `debate.early_stop_consensus` | `true` | Adaptive stopping |
| `evaluation.self_consistency_samples` | `6` | SC baseline samples |

---

## Baselines

| Method | Description |
|--------|-------------|
| **Direct QA** | Single model, zero-shot CoT. No debate. |
| **Self-Consistency** | N=6 samples (matching total LLM calls in a 3-round debate), majority vote. |
| **Debate Pipeline** | Full 3-round debate + judge verdict. |

---

## LLM Usage Disclosure

This project used Claude (claude-sonnet-4-20250514) as the debate agents and judge. The Streamlit UI was built with AI assistance. All code logic, prompt design, and analysis in the blog post are original.

---

## Key References

1. Irving, G., Christiano, P., & Amodei, D. (2018). *AI Safety via Debate*. arXiv:1805.00899.
2. Liang, T. et al. (2024). *Encouraging Divergent Thinking in LLMs through Multi-Agent Debate*. EMNLP 2024.
3. Kenton, Z. et al. (2024). *On Scalable Oversight with Weak LLMs Judging Strong LLMs*. NeurIPS 2024.
4. Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in LLMs*. NeurIPS 2022.
5. Wang, X. et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in LLMs*. ICLR 2023.
