# Multi-Agent Hallucination Detection System

##  Execution Instructions

To run the complete multi-agent pipeline, the agents must be executed in a **specific order**, as each agent produces features required by the next one.

### Important

Make sure all paths are correctly set (e.g., `/content/drive/MyDrive/hallucination_nlp/`) and that your environment (Google Colab or local) has access to the required data  or use the files provides in data folder

---

## Execution Order

Run the following scripts **in this exact order**:

### 1. Semantic Agent (Embedding-based similarity)

```bash
!python /content/drive/MyDrive/hallucination_nlp/agent_sementic.py
 py agent_sementic.py
```

###  NLI Agent (Semantic reasoning)

```bash
!python /content/drive/MyDrive/hallucination_nlp/agent_nli.py
py agent_nli.py
```

### 3. Arbitration Agent (Feature fusion + final classification)

```bash
!python /content/drive/MyDrive/hallucination_nlp/agent_arbitre.py
py agent_arbitre.py
```

### 4. Span Detection Agent (Hallucination localization)

```bash
!python /content/drive/MyDrive/hallucination_nlp/agent_spam.py

py agent_spam.py
```

---

##  Pipeline Description

* **Semantic Agent**: Computes similarity scores using embeddings.
* **NLI Agent**: Evaluates logical consistency (entailment, contradiction, neutral).
* **Arbitration Agent**: Combines all features and produces final predictions.
* **Span Agent**: Detects hallucinated segments within the text.

---

##Notes

* The order is **critical**: each agent depends on outputs from previous steps.
* Outputs are saved as intermediate `.jsonl` files.
* GPU is recommended for faster execution.

---

## Expected Outcome

After running all agents, you will obtain:

* Final hallucination predictions
* Feature files for analysis
* Span-level hallucination detection
