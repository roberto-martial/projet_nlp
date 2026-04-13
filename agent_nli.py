"""
AGENT NLI 
"""

import json
import os
import re
import numpy as np
import torch
from transformers import pipeline
from sklearn.metrics import classification_report, roc_auc_score


# ─────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────

def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                obj = json.loads(obj)
            records.append(obj)
    return records


def split_context(context: str):
    parts = context.split("[SEP]")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return context.strip(), ""


def split_sentences(text: str):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def truncate(text: str, max_chars=300):
    return text[:max_chars] if len(text) > max_chars else text


# ─────────────────────────────────────────────
# AGENT NLI
# ─────────────────────────────────────────────

class NLIAgent:

    # Mapping connu pour cross-encoder/nli-distilroberta-base
    # LABEL_0=contradiction, LABEL_1=entailment, LABEL_2=neutral
    LABEL_MAP = {
        "label_0": "contradiction",
        "label_1": "entailment",
        "label_2": "neutral",
        # Au cas où le modèle retourne déjà les bons noms
        "contradiction": "contradiction",
        "entailment":    "entailment",
        "neutral":       "neutral",
    }

    def __init__(
        self,
        model_name="cross-encoder/nli-distilroberta-base",
        threshold=0.5,
        batch_size=32
    ):
        device = 0 if torch.cuda.is_available() else -1
        print(f"Chargement modèle NLI sur device={device}")

        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None, 
            device=device
        )

        self.threshold  = threshold
        self.batch_size = batch_size

        # ── Debug : afficher le format brut retourné par le pipeline ──
        self._debug_output_format()

        print("Modèle prêt\n")

    def _debug_output_format(self):
        """
        Fait une inférence test et affiche le format brut
        pour s'assurer que le parsing est correct.
        """
        test_text = "The sky is blue </s> The sky is blue"
        raw = self.classifier(test_text)
        print(f"\n Format brut du pipeline :")
        print(f"   type(raw)    = {type(raw)}")
        print(f"   type(raw[0]) = {type(raw[0])}")
        print(f"   raw[0]       = {raw[0]}")

    def _parse_output(self, raw_output) -> dict:
        """
        Parse un output du pipeline en dict {label: score}.
        Gère les deux formats possibles :
          - Liste de dicts : [{"label": "LABEL_0", "score": 0.9}, ...]
          - Dict direct    : {"label": "LABEL_0", "score": 0.9}
        """
        scores = {}

        # Cas 1 : liste de dicts (format standard return_all_scores=True)
        if isinstance(raw_output, list):
            for item in raw_output:
                if isinstance(item, dict) and "label" in item and "score" in item:
                    key = self.LABEL_MAP.get(item["label"].lower(), item["label"].lower())
                    scores[key] = item["score"]

        # Cas 2 : dict unique
        elif isinstance(raw_output, dict) and "label" in raw_output:
            key = self.LABEL_MAP.get(raw_output["label"].lower(), raw_output["label"].lower())
            scores[key] = raw_output["score"]

        return scores

    def _batch_classify(self, premises, hypotheses) -> list[dict]:
        """
        Classifie des paires (premise, hypothesis) en batch.
        Retourne une liste de dicts {entailment, neutral, contradiction}.
        """
        texts   = [p + " </s> " + h for p, h in zip(premises, hypotheses)]
        outputs = self.classifier(texts, batch_size=self.batch_size)

        results = []
        for out in outputs:
            scores = self._parse_output(out)
            results.append(scores)

        return results

    def compute_features(self, entries, verbose=True) -> list[dict]:
        """
        Calcule les features NLI pour chaque entrée.

        Features :
          nli_entailment_max        : max entailment sur toutes les phrases
          nli_contradiction_max     : max contradiction  ← signal principal
          nli_neutral_avg           : neutral moyen
          nli_entailment_avg        : entailment moyen
          nli_contradiction_avg     : contradiction moyenne
          nli_contradiction_weighted: contradiction pondérée par longueur
          nli_n_sentences           : nb phrases analysées
        """
        all_features = []
        total = len(entries)

        for i, entry in enumerate(entries):

            if verbose and i % 200 == 0:
                print(f"  [{i}/{total}] processing...")

            knowledge, _ = split_context(entry["context"])
            knowledge     = truncate(knowledge, 300)

            sentences = split_sentences(entry["response"])
            if not sentences:
                sentences = [truncate(entry["response"], 200)]

            premises   = [knowledge] * len(sentences)
            hypotheses = [truncate(s, 200) for s in sentences]

            try:
                results = self._batch_classify(premises, hypotheses)
            except Exception as e:
                print(f"    Erreur entrée {i}: {e}")
                all_features.append(self._default_features())
                continue

            entailments    = []
            neutrals       = []
            contradictions = []
            weights        = []

            for sentence, scores in zip(sentences, results):
                entailments.append(scores.get("entailment", 0.33))
                neutrals.append(scores.get("neutral", 0.33))
                contradictions.append(scores.get("contradiction", 0.33))
                weights.append(max(len(sentence), 1))

            if not contradictions:
                all_features.append(self._default_features())
                continue

            all_features.append({
                "nli_entailment_max":         float(np.max(entailments)),
                "nli_contradiction_max":       float(np.max(contradictions)),
                "nli_neutral_avg":             float(np.mean(neutrals)),
                "nli_entailment_avg":          float(np.mean(entailments)),
                "nli_contradiction_avg":       float(np.mean(contradictions)),
                "nli_contradiction_weighted":  float(np.average(contradictions, weights=weights)),
                "nli_n_sentences":             len(sentences),
            })

        return all_features

    def _default_features(self):
        return {
            "nli_entailment_max":         0.33,
            "nli_contradiction_max":      0.33,
            "nli_neutral_avg":            0.33,
            "nli_entailment_avg":         0.33,
            "nli_contradiction_avg":      0.33,
            "nli_contradiction_weighted": 0.33,
            "nli_n_sentences":            0,
        }

    def predict(self, features):
        scores = np.array([f["nli_contradiction_max"] for f in features])
        return (scores > self.threshold).astype(int)

    def find_best_threshold(self, features, labels):
        scores  = np.array([f["nli_contradiction_max"] for f in features])
        best_t, best_f1 = self.threshold, 0.0

        for t in np.arange(0.1, 0.9, 0.02):
            preds     = (scores > t).astype(int)
            tp        = np.sum((preds == 1) & (labels == 1))
            fp        = np.sum((preds == 1) & (labels == 0))
            fn        = np.sum((preds == 0) & (labels == 1))
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        return float(best_t), float(best_f1)


# ─────────────────────────────────────────────
# ÉVALUATION
# ─────────────────────────────────────────────

def evaluate(labels, preds, features, split_name=""):
    print("\n" + "="*50)
    print(f" NLI — {split_name.upper()}")
    print("="*50)
    print(classification_report(labels, preds, target_names=["Correct", "Halluciné"]))
    try:
        scores = np.array([f["nli_contradiction_max"] for f in features])
        auc    = roc_auc_score(labels, scores)
        print(f"AUC-ROC: {auc:.4f}")
    except Exception as e:
        print(f"AUC error: {e}")


# ─────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────

def save_features(entries, features, preds, path):
    with open(path, "w", encoding="utf-8") as f:
        for e, feat, pred in zip(entries, features, preds):
            f.write(json.dumps({
                "id":       e["id"],
                "label":    e["label"],
                "source":   e["source"],
                **feat,
                "nli_pred": int(pred)
            }, ensure_ascii=False) + "\n")
    print(f" Saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    BASE       = "/content/drive/MyDrive/hallucination_nlp"
    DATA_DIR   = os.path.join(BASE, "data/unified")
    OUTPUT_DIR = os.path.join(BASE, "data/agent_features")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(" Loading data...")
    val_data  = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    test_data = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))
    print(f"  Val  : {len(val_data)} entrées")
    print(f"  Test : {len(test_data)} entrées")

    agent = NLIAgent()

    # ── VAL ──
    print("\n VAL...")
    val_labels   = np.array([e["label"] for e in val_data])
    val_features = agent.compute_features(val_data, verbose=True)

    best_t, best_f1 = agent.find_best_threshold(val_features, val_labels)
    print(f"\n Best threshold: {best_t:.2f} | F1={best_f1:.4f}")

    agent.threshold = best_t
    val_preds = agent.predict(val_features)
    evaluate(val_labels, val_preds, val_features, "val")

    # ── TEST ──
    print("\n TEST...")
    test_labels   = np.array([e["label"] for e in test_data])
    test_features = agent.compute_features(test_data, verbose=True)
    test_preds    = agent.predict(test_features)
    evaluate(test_labels, test_preds, test_features, "test")

    # ── SAVE ──
    save_features(val_data,  val_features, val_preds,
                  os.path.join(OUTPUT_DIR, "nli_val.jsonl"))
    save_features(test_data, test_features, test_preds,
                  os.path.join(OUTPUT_DIR, "nli_test.jsonl"))

    print("\n Agent NLI terminé !")