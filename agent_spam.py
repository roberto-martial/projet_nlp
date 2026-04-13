"""
ÉTAPE 4 — Agent Span (Détecteur de Spans Hallucinés)

"""

import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score
from collections import defaultdict


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

LABEL2ID = {"O": 0, "B-HAL": 1, "I-HAL": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 256
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5


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


# ─────────────────────────────────────────────
# CONSTRUCTION DES LABELS BIO
# ─────────────────────────────────────────────

def build_bio_labels(text: str, spans: list[dict]) -> list[tuple[str, str]]:
    """
    Construit une liste de (mot, label_BIO) pour un texte et ses spans.

    Exemple :
      text  = "Jake Gyllenhaal starred with Tom Hanks"
      spans = [{"start": 28, "end": 37, "text": "Tom Hanks"}]
      →  [("Jake","O"), ("Gyllenhaal","O"), ("starred","O"),
          ("with","O"), ("Tom","B-HAL"), ("Hanks","I-HAL")]
    """
    # Marquer les positions de caractères hallucinées
    char_labels = ["O"] * len(text)
    for span in spans:
        start = span.get("start", 0)
        end   = span.get("end",   0)
        for i in range(start, min(end, len(text))):
            char_labels[i] = "HAL"

    # Tokenisation naïve par espaces → on aligne les labels
    tokens     = []
    bio_labels = []
    i          = 0

    while i < len(text):
        # Sauter les espaces
        if text[i] == " ":
            i += 1
            continue

        # Trouver la fin du mot
        j = i
        while j < len(text) and text[j] != " ":
            j += 1

        word        = text[i:j]
        word_chars  = char_labels[i:j]

        # Déterminer le label BIO du mot
        if "HAL" in word_chars:
            # Premier token HAL du span → B-HAL, suivants → I-HAL
            if not tokens or bio_labels[-1] == "O":
                label = "B-HAL"
            else:
                label = "I-HAL"
        else:
            label = "O"

        tokens.append(word)
        bio_labels.append(label)
        i = j

    return list(zip(tokens, bio_labels))


# ─────────────────────────────────────────────
# DATASET PYTORCH
# ─────────────────────────────────────────────

class SpanDataset(Dataset):

    def __init__(self, entries: list[dict], tokenizer, max_len: int = MAX_LEN):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.samples   = self._prepare(entries)

    def _prepare(self, entries: list[dict]) -> list[dict]:
        samples = []

        for entry in entries:
            response = entry["response"]
            spans    = entry.get("spans", [])

            # Construire les labels BIO mot par mot
            word_label_pairs = build_bio_labels(response, spans)

            if not word_label_pairs:
                continue

            words  = [w for w, _ in word_label_pairs]
            labels = [LABEL2ID[l] for _, l in word_label_pairs]

            # Tokenisation sub-word avec alignement des labels
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            # Aligner les labels sur les sub-tokens
            word_ids      = encoding.word_ids(batch_index=0)
            aligned_labels = []
            prev_word_id   = None

            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)   # tokens spéciaux ignorés
                elif word_id != prev_word_id:
                    aligned_labels.append(labels[word_id])
                else:
                    # Sub-token suivant : I-HAL si le mot est halluciné
                    aligned_labels.append(
                        LABEL2ID["I-HAL"] if labels[word_id] in [LABEL2ID["B-HAL"], LABEL2ID["I-HAL"]]
                        else labels[word_id]
                    )
                prev_word_id = word_id

            samples.append({
                "input_ids":      encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels":         torch.tensor(aligned_labels, dtype=torch.long),
                "meta": {
                    "id":     entry["id"],
                    "source": entry["source"],
                    "label":  entry["label"],
                }
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "input_ids":      s["input_ids"],
            "attention_mask": s["attention_mask"],
            "labels":         s["labels"],
        }


# ─────────────────────────────────────────────
# AGENT SPAN
# ─────────────────────────────────────────────

class SpanAgent:

    def __init__(self, model_name: str = MODEL_NAME):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True
        ).to(self.device)
        print(f" SpanAgent prêt sur {self.device}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        optimizer = AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps
        )

        best_val_f1 = 0.0

        for epoch in range(EPOCHS):
            # ── Train ──
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_loss = total_loss / len(train_loader)

            # ── Val ──
            val_f1 = self._evaluate_loader(val_loader)
            print(f"  Epoch {epoch+1}/{EPOCHS} | Loss={avg_loss:.4f} | Val F1-HAL={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), "/tmp/best_span_model.pt")
                print(f"   Meilleur modèle sauvegardé (F1={best_val_f1:.4f})")

        # Recharger le meilleur modèle
        self.model.load_state_dict(torch.load("/tmp/best_span_model.pt"))
        print(f"\n Entraînement terminé | Meilleur Val F1-HAL={best_val_f1:.4f}")

    def _evaluate_loader(self, loader: DataLoader) -> float:
        """Évalue le F1 sur les tokens HAL (B-HAL + I-HAL)."""
        self.model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                preds = torch.argmax(outputs.logits, dim=-1)

                for pred_seq, label_seq in zip(preds, labels):
                    for p, l in zip(pred_seq, label_seq):
                        if l.item() == -100:
                            continue
                        all_preds.append(p.item())
                        all_labels.append(l.item())

        # F1 sur les classes HAL uniquement (1 et 2)
        f1 = f1_score(all_labels, all_preds, labels=[1, 2], average="macro", zero_division=0)
        return f1

    def predict_entry(self, text: str) -> list[dict]:
        """
        Prédit les spans hallucinés pour un texte donné.
        Retourne une liste de spans {"start", "end", "text", "confidence"}.
        """
        words    = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=MAX_LEN,
            truncation=True,
            return_tensors="pt"
        )

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding["input_ids"].to(self.device),
                attention_mask=encoding["attention_mask"].to(self.device)
            )

        logits   = outputs.logits[0]
        probs    = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = np.argmax(probs, axis=-1)
        word_ids = encoding.word_ids(batch_index=0)

        # Reconstruire les spans depuis les prédictions
        word_preds = {}
        for token_idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id not in word_preds:
                word_preds[word_id] = {
                    "label": ID2LABEL[pred_ids[token_idx]],
                    "conf":  float(probs[token_idx][pred_ids[token_idx]])
                }

        # Convertir les prédictions mot→span en positions de caractères
        spans     = []
        char_pos  = 0
        in_span   = False
        span_start = 0
        span_conf  = []

        for word_id, word in enumerate(words):
            pred = word_preds.get(word_id, {"label": "O", "conf": 1.0})

            if pred["label"] in ["B-HAL", "I-HAL"]:
                if not in_span:
                    span_start = char_pos
                    in_span    = True
                    span_conf  = []
                span_conf.append(pred["conf"])
            else:
                if in_span:
                    spans.append({
                        "start":      span_start,
                        "end":        char_pos - 1,
                        "text":       text[span_start:char_pos - 1],
                        "confidence": round(float(np.mean(span_conf)), 4)
                    })
                    in_span = False

            char_pos += len(word) + 1  # +1 pour l'espace

        # Fermer le dernier span si nécessaire
        if in_span:
            spans.append({
                "start":      span_start,
                "end":        char_pos - 1,
                "text":       text[span_start:char_pos - 1],
                "confidence": round(float(np.mean(span_conf)), 4)
            })

        return spans

    def evaluate_test(self, test_loader: DataLoader, split_name: str = "test"):
        """Évaluation complète avec rapport de classification."""
        self.model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                preds = torch.argmax(outputs.logits, dim=-1)

                for pred_seq, label_seq in zip(preds, labels):
                    for p, l in zip(pred_seq, label_seq):
                        if l.item() == -100:
                            continue
                        all_preds.append(p.item())
                        all_labels.append(l.item())

        print(f"\n{'='*50}")
        print(f" SPAN AGENT — {split_name.upper()}")
        print(f"{'='*50}")
        print(classification_report(
            all_labels, all_preds,
            target_names=["O", "B-HAL", "I-HAL"],
            zero_division=0
        ))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    BASE       = "/content/drive/MyDrive/hallucination_nlp"
    DATA_DIR   = os.path.join(BASE, "data/unified")
    OUTPUT_DIR = os.path.join(BASE, "data/agent_features")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Chargement ──
    print(" Chargement des données...")
    train_data = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    val_data   = load_jsonl(os.path.join(DATA_DIR, "val.jsonl"))
    test_data  = load_jsonl(os.path.join(DATA_DIR, "test.jsonl"))

    # ── Filtrer uniquement les entrées avec spans annotés ──
    # Seul general_data a des spans → on filtre sur source="general"
    train_span = [e for e in train_data if e["source"] == "general"]
    val_span   = [e for e in val_data   if e["source"] == "general"]
    test_span  = [e for e in test_data  if e["source"] == "general"]

    print(f"  Train span : {len(train_span)} | Val : {len(val_span)} | Test : {len(test_span)}")

    # ── Tokenizer & Datasets ──
    print("\n Préparation des datasets...")
    tokenizer    = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = SpanDataset(train_span, tokenizer)
    val_dataset   = SpanDataset(val_span,   tokenizer)
    test_dataset  = SpanDataset(test_span,  tokenizer)

    print(f"  Train : {len(train_dataset)} | Val : {len(val_dataset)} | Test : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Entraînement ──
    print("\n  Entraînement du SpanAgent...")
    agent = SpanAgent()
    agent.train(train_loader, val_loader)

    # ── Évaluation finale ──
    agent.evaluate_test(test_loader, "test")

    # ── Démonstration sur quelques exemples ──
    print("\n EXEMPLES DE SPANS DÉTECTÉS")
    print("="*50)
    for entry in test_span[:3]:
        spans = agent.predict_entry(entry["response"])
        label = " HALLUCINÉ" if entry["label"] == 1 else " CORRECT"
        print(f"\n[{label}]")
        print(f"  Réponse : {entry['response'][:150]}")
        print(f"  Vrais spans   : {entry.get('spans', [])}")
        print(f"  Spans détectés: {spans}")

    print("\n Agent Span terminé !")