import json
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, roc_auc_score
from collections import defaultdict




def split_context(context: str):
    parts = context.split("[SEP]")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return context, ""


def split_sentences(text: str):
    return [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]




class SemanticAgent:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", threshold=0.4):
        print(f"⏳ Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        print(" Model loaded\n")

    def encode(self, texts):
        return self.model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def compute_features(self, contexts, responses):
        """
        Retourne un dict de features pour chaque entrée
        """

        knowledge_texts = []
        query_texts = []

        for ctx in contexts:
            k, q = split_context(ctx)
            knowledge_texts.append(k)
            query_texts.append(q if q else k)

        print(" Encoding knowledge...")
        emb_k = self.encode(knowledge_texts)

        print(" Encoding query...")
        emb_q = self.encode(query_texts)

        print(" Encoding responses...")
        emb_r = self.encode(responses)

        # Scores globaux
        score_k = np.sum(emb_k * emb_r, axis=1)
        score_q = np.sum(emb_q * emb_r, axis=1)

        # Score combiné (pondéré)
        combined = 0.7 * score_k + 0.3 * score_q

        # Analyse phrase par phrase
        min_scores = []
        avg_scores = []

        print(" Sentence-level analysis...")
        for i, resp in enumerate(responses):
            sentences = split_sentences(resp)

            if not sentences:
                min_scores.append(combined[i])
                avg_scores.append(combined[i])
                continue

            sent_emb = self.encode(sentences)
            k_vec = emb_k[i]

            scores = np.dot(sent_emb, k_vec)

            min_scores.append(float(np.min(scores)))
            avg_scores.append(float(np.mean(scores)))

        features = []
        for i in range(len(contexts)):
            features.append({
                "score_knowledge": float(score_k[i]),
                "score_query": float(score_q[i]),
                "score_combined": float(combined[i]),
                "score_min_sentence": float(min_scores[i]),
                "score_avg_sentence": float(avg_scores[i]),
                "divergence": float(1 - combined[i])
            })

        return features

    def predict(self, features):
        scores = np.array([f["score_combined"] for f in features])
        return (scores > self.threshold).astype(int)

    def find_best_threshold(self, features, labels):
        scores = np.array([f["score_combined"] for f in features])

        best_t, best_f1 = self.threshold, 0

        for t in np.arange(0.1, 0.9, 0.02):
            preds = (scores > t).astype(int) 

            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_f1, best_t = f1, t

        return best_t, best_f1



# ÉVALUATION


def evaluate(labels, preds, scores):
    print(classification_report(labels, preds))

    try:
        auc = roc_auc_score(labels, scores)
        print(f"AUC: {auc:.4f}")
    except:
        pass



# SAUVEGARDE


def save_features(entries, features, preds, path):
    with open(path, "w", encoding="utf-8") as f:
        for e, feat, p in zip(entries, features, preds):
            f.write(json.dumps({
                "id": e["id"],
                "label": e["label"],
                "source": e["source"],
                **feat,
                "semantic_pred": int(p)
            }) + "\n")



# MAIN


if __name__ == "__main__":

    BASE_PROJECT = "/content/drive/MyDrive/hallucination_nlp"
    DATA_DIR = os.path.join(BASE_PROJECT, "data/unified")
    OUTPUT_DIR = os.path.join(BASE_PROJECT, "data/agent_features")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    val_data = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    test_data = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    agent = SemanticAgent()

    # ---------- VAL ----------
    val_ctx = [e["context"] for e in val_data]
    val_resp = [e["response"] for e in val_data]
    val_labels = np.array([e["label"] for e in val_data])

    val_features = agent.compute_features(val_ctx, val_resp)

    best_t, best_f1 = agent.find_best_threshold(val_features, val_labels)
    print(f"\n Best threshold: {best_t:.2f} | F1={best_f1:.4f}")

    agent.threshold = best_t
    val_preds = agent.predict(val_features)

    val_scores = np.array([f["score_combined"] for f in val_features])
    evaluate(val_labels, val_preds, val_scores)

    # ---------- TEST ----------
    test_ctx = [e["context"] for e in test_data]
    test_resp = [e["response"] for e in test_data]
    test_labels = np.array([e["label"] for e in test_data])

    test_features = agent.compute_features(test_ctx, test_resp)
    test_preds = agent.predict(test_features)

    test_scores = np.array([f["score_combined"] for f in test_features])
    evaluate(test_labels, test_preds, test_scores)

    # ---------- SAVE ----------
    save_features(val_data, val_features, val_preds,
                  os.path.join(OUTPUT_DIR, "semantic_val.jsonl"))

    save_features(test_data, test_features, test_preds,
                  os.path.join(OUTPUT_DIR, "semantic_test.jsonl"))

    print("\n Semantic Agent ready for Meta-Agent!")