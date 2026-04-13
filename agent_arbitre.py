"""
ÉTAPE 4 — Agent Arbitre (Meta-Agent)
======================================
Objectif : combiner les features des agents précédents
           pour produire une décision finale plus robuste.

Input  : features sémantiques + NLI sauvegardées en jsonl
Output : label binaire (halluciné / correct) + score de confiance

Modèles testés :
  - LogisticRegression  (baseline rapide)
  - RandomForest        (capture non-linéarités)
  - GradientBoosting    (meilleur en général sur features tabulaires)
"""

import json
import os
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
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


def merge_features(semantic_data: list[dict], nli_data: list[dict]) -> dict:
    """
    Fusionne les features sémantiques et NLI par ID.
    Retourne un dict {id: merged_entry}.
    """
    merged = {}

    for entry in semantic_data:
        merged[entry["id"]] = {
            "id":     entry["id"],
            "label":  entry["label"],
            "source": entry["source"],
            # Features sémantiques
            "score_knowledge":    entry.get("score_knowledge", 0.0),
            "score_query":        entry.get("score_query", 0.0),
            "score_combined":     entry.get("score_combined", 0.0),
            "score_min_sentence": entry.get("score_min_sentence", 0.0),
            "score_avg_sentence": entry.get("score_avg_sentence", 0.0),
            "divergence":         entry.get("divergence", 0.5),
        }

    for entry in nli_data:
        if entry["id"] in merged:
            merged[entry["id"]].update({
                # Features NLI
                "nli_entailment_max":         entry.get("nli_entailment_max", 0.33),
                "nli_contradiction_max":       entry.get("nli_contradiction_max", 0.33),
                "nli_neutral_avg":             entry.get("nli_neutral_avg", 0.33),
                "nli_entailment_avg":          entry.get("nli_entailment_avg", 0.33),
                "nli_contradiction_avg":       entry.get("nli_contradiction_avg", 0.33),
                "nli_contradiction_weighted":  entry.get("nli_contradiction_weighted", 0.33),
                "nli_n_sentences":             entry.get("nli_n_sentences", 0),
                # Prédictions individuelles des agents
                "semantic_pred":               entry.get("semantic_pred", 0),
                "nli_pred":                    entry.get("nli_pred", 0),
            })

    return merged


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # Agent sémantique
    "score_knowledge",
    "score_query",
    "score_combined",
    "score_min_sentence",
    "score_avg_sentence",
    "divergence",
    # Agent NLI
    "nli_entailment_max",
    "nli_contradiction_max",
    "nli_neutral_avg",
    "nli_entailment_avg",
    "nli_contradiction_avg",
    "nli_contradiction_weighted",
    "nli_n_sentences",
    # Features croisées (interaction entre agents)
    "agent_agreement",        # les deux agents sont-ils d'accord ?
    "combined_halluc_score",  # score composite d'hallucination
]


def build_feature_matrix(merged: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Construit la matrice de features X et le vecteur labels y.
    Ajoute des features croisées inter-agents.
    """
    entries = list(merged.values())
    X_list  = []
    y_list  = []

    for e in entries:
        # Feature croisée 1 : accord entre les deux agents
        agent_agreement = 1.0 if e.get("semantic_pred", 0) == e.get("nli_pred", 0) else 0.0

        # Feature croisée 2 : score composite hallucination
        # divergence élevée + contradiction élevée = très suspect
        combined_halluc = (
            0.5 * e.get("divergence", 0.5) +
            0.5 * e.get("nli_contradiction_max", 0.33)
        )

        row = [
            e.get("score_knowledge",           0.0),
            e.get("score_query",               0.0),
            e.get("score_combined",            0.0),
            e.get("score_min_sentence",        0.0),
            e.get("score_avg_sentence",        0.0),
            e.get("divergence",                0.5),
            e.get("nli_entailment_max",        0.33),
            e.get("nli_contradiction_max",     0.33),
            e.get("nli_neutral_avg",           0.33),
            e.get("nli_entailment_avg",        0.33),
            e.get("nli_contradiction_avg",     0.33),
            e.get("nli_contradiction_weighted",0.33),
            e.get("nli_n_sentences",           0),
            agent_agreement,
            combined_halluc,
        ]

        X_list.append(row)
        y_list.append(e["label"])

    return np.array(X_list), np.array(y_list), entries


# ─────────────────────────────────────────────
# AGENT ARBITRE
# ─────────────────────────────────────────────

class MetaAgent:

    def __init__(self):
        self.models = {
            "LogisticRegression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    C=1.0
                ))
            ]),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                random_state=42
            ),
        }
        self.best_model_name = None
        self.best_model      = None

    def train_and_select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray
    ):
        """
        Entraîne tous les modèles sur train,
        évalue sur val, sélectionne le meilleur par AUC.
        """
        print("\n" + "="*50)
        print("  ENTRAÎNEMENT DES MODÈLES")
        print("="*50)

        best_auc = 0.0

        for name, model in self.models.items():
            print(f"\n   {name}...")
            model.fit(X_train, y_train)

            val_proba = model.predict_proba(X_val)[:, 1]
            val_preds = model.predict(X_val)
            auc       = roc_auc_score(y_val, val_proba)

            tp = np.sum((val_preds == 1) & (y_val == 1))
            fp = np.sum((val_preds == 1) & (y_val == 0))
            fn = np.sum((val_preds == 0) & (y_val == 1))
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)

            print(f"     AUC={auc:.4f} | F1={f1:.4f} | P={precision:.4f} | R={recall:.4f}")

            if auc > best_auc:
                best_auc             = auc
                self.best_model_name = name
                self.best_model      = model

        print(f"\n Meilleur modèle : {self.best_model_name} (AUC={best_auc:.4f})")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = self.best_model.predict(X)
        proba = self.best_model.predict_proba(X)[:, 1]
        return preds, proba

    def feature_importance(self, top_n: int = 10):
        """Affiche l'importance des features pour le meilleur modèle."""
        clf = self.best_model
        # Extraire le classifieur réel si Pipeline
        if hasattr(clf, "named_steps"):
            clf = clf.named_steps["clf"]

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = np.abs(clf.coef_[0])
        else:
            print("  Feature importance non disponible pour ce modèle.")
            return

        indices = np.argsort(importances)[::-1][:top_n]
        print(f"\n Top {top_n} features ({self.best_model_name}) :")
        for rank, i in enumerate(indices, 1):
            print(f"  {rank:>2}. {FEATURE_COLS[i]:<35} {importances[i]:.4f}")


# ─────────────────────────────────────────────
# ÉVALUATION
# ─────────────────────────────────────────────

def evaluate(labels, preds, proba, split_name="", entries=None):
    print(f"\n{'='*50}")
    print(f" ARBITRE — {split_name.upper()}")
    print(f"{'='*50}")
    print(classification_report(labels, preds, target_names=["Correct", "Halluciné"]))

    try:
        auc = roc_auc_score(labels, proba)
        print(f"  AUC-ROC : {auc:.4f}")
    except Exception as e:
        print(f"  AUC error: {e}")

    # Décomposition par source
    if entries:
        print(f"\n  Par source :")
        groups = defaultdict(lambda: {"labels": [], "preds": [], "proba": []})
        for e, p, pb in zip(entries, preds, proba):
            src = e["source"]
            groups[src]["labels"].append(e["label"])
            groups[src]["preds"].append(p)
            groups[src]["proba"].append(pb)

        for src, data in sorted(groups.items()):
            lbls = np.array(data["labels"])
            prds = np.array(data["preds"])
            prbs = np.array(data["proba"])
            try:
                auc_src = roc_auc_score(lbls, prbs)
            except Exception:
                auc_src = float("nan")
            tp = np.sum((prds == 1) & (lbls == 1))
            fp = np.sum((prds == 1) & (lbls == 0))
            fn = np.sum((prds == 0) & (lbls == 1))
            p_ = tp / (tp + fp + 1e-8)
            r_ = tp / (tp + fn + 1e-8)
            f1 = 2 * p_ * r_ / (p_ + r_ + 1e-8)
            print(f"    [{src:<16}] AUC={auc_src:.3f} | F1={f1:.3f} | n={len(lbls)}")


# ─────────────────────────────────────────────
# SAUVEGARDE
# ─────────────────────────────────────────────

def save_predictions(entries, preds, proba, path):
    with open(path, "w", encoding="utf-8") as f:
        for e, pred, prob in zip(entries, preds, proba):
            f.write(json.dumps({
                "id":              e["id"],
                "label":           e["label"],
                "source":          e["source"],
                "final_pred":      int(pred),
                "confidence":      round(float(prob), 4),
                "correct":         int(pred) == e["label"],
            }, ensure_ascii=False) + "\n")
    print(f"💾 Saved → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    BASE       = "/content/drive/MyDrive/hallucination_nlp"
    FEAT_DIR   = os.path.join(BASE, "data/agent_features")
    OUTPUT_DIR = os.path.join(BASE, "data/arbitre_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Chargement des features ──
    print(" Chargement des features...")

    train_sem = load_jsonl(os.path.join(FEAT_DIR, "semantic_val.jsonl"))
    train_nli = load_jsonl(os.path.join(FEAT_DIR, "nli_val.jsonl"))
    test_sem  = load_jsonl(os.path.join(FEAT_DIR, "semantic_test.jsonl"))
    test_nli  = load_jsonl(os.path.join(FEAT_DIR, "nli_test.jsonl"))

    print(f"  Train sémantique : {len(train_sem)} | Train NLI : {len(train_nli)}")
    print(f"  Test  sémantique : {len(test_sem)}  | Test  NLI : {len(test_nli)}")

    # ── Fusion des features ──
    print("\n Fusion des features...")
    train_merged = merge_features(train_sem, train_nli)
    test_merged  = merge_features(test_sem,  test_nli)
    print(f"  Train fusionné : {len(train_merged)} entrées")
    print(f"  Test  fusionné : {len(test_merged)} entrées")

    # ── Construction des matrices ──
    X_train, y_train, train_entries = build_feature_matrix(train_merged)
    X_test,  y_test,  test_entries  = build_feature_matrix(test_merged)
    print(f"\n  Shape X_train : {X_train.shape}")
    print(f"  Shape X_test  : {X_test.shape}")
    print(f"  Features      : {FEATURE_COLS}")

    # ── Entraînement et sélection ──
    # On utilise 80% du train pour fit, 20% pour la sélection du modèle
    n_val  = int(len(X_train) * 0.2)
    X_tr   = X_train[n_val:]
    y_tr   = y_train[n_val:]
    X_val  = X_train[:n_val]
    y_val  = y_train[:n_val]

    meta = MetaAgent()
    meta.train_and_select(X_tr, y_tr, X_val, y_val)

    # ── Feature importance ──
    meta.feature_importance(top_n=10)

    # ── Évaluation finale sur test ──
    test_preds, test_proba = meta.predict(X_test)
    evaluate(y_test, test_preds, test_proba, "test", test_entries)

    # ── Sauvegarde ──
    save_predictions(test_entries, test_preds, test_proba,
                     os.path.join(OUTPUT_DIR, "final_predictions.jsonl"))

    print("\n Agent Arbitre terminé !")