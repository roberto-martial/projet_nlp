import json
import os
import re
from typing import Any
from collections import Counter
import random




def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def find_span_positions(text: str, span: str) -> list[dict]:
    pattern = re.escape(span.strip())
    matches = re.finditer(pattern, text, re.IGNORECASE)

    return [
        {
            "start": m.start(),
            "end": m.end(),
            "text": text[m.start():m.end()]
        }
        for m in matches
    ]


def build_context(knowledge: str = "", dialogue_history: str = "") -> str:
    knowledge = clean_text(knowledge)
    dialogue_history = clean_text(dialogue_history)

    if knowledge and dialogue_history:
        return f"{knowledge} [SEP] {dialogue_history}"
    elif knowledge:
        return knowledge
    elif dialogue_history:
        return dialogue_history
    else:
        return ""


def make_id(source: str, index: int, label: int) -> str:
    suffix = "hallucinated" if label == 1 else "correct"
    return f"{source}_{index:04d}_{suffix}"

#ici on commence avec dialogue_data

def parse_dialogue_data(raw: dict, index: int) -> list[dict]:
    entries = []

    context = build_context(
        raw.get("knowledge", ""),
        raw.get("dialogue_history", "")
    )

    for label, key in [(0, "right_response"), (1, "hallucinated_response")]:
        response = clean_text(raw.get(key, ""))

        if not response:
            continue

        entries.append({
            "id": make_id("dialogue", index, label),
            "source": "dialogue",
            "context": context,
            "response": response,
            "label": label,
            "spans": [],
            "meta": {
                "context_len": len(context),
                "response_len": len(response)
            }
        })

    return entries


def parse_general_data(raw: dict, index: int) -> list[dict]:
    response = clean_text(raw.get("chatgpt_response", ""))
    label = 1 if raw.get("hallucination", "no").lower() == "yes" else 0

    spans = []
    if label == 1:
        for span_text in raw.get("hallucination_spans", []):
            spans.extend(find_span_positions(response, span_text))

    context = f"Query: {clean_text(raw.get('user_query', ''))}"

    return [{
        "id": make_id("general", index, label),
        "source": "general",
        "context": context,
        "response": response,
        "label": label,
        "spans": spans,
        "meta": {
            "context_len": len(context),
            "response_len": len(response)
        }
    }]


def parse_qa_data(raw: dict, index: int) -> list[dict]:
    entries = []

    context = build_context(
    knowledge=raw.get("knowledge", ""),
    dialogue_history=raw.get("question", "")
)

    for label, key in [(0, "right_answer"), (1, "hallucinated_answer")]:
        response = clean_text(raw.get(key, ""))

        if not response:
            continue

        entries.append({
            "id": make_id("qa", index, label),
            "source": "qa",
            "context": context,
            "response": response,
            "label": label,
            "spans": [],
            "meta": {
                "context_len": len(context),
                "response_len": len(response)
            }
        })

    return entries


def parse_summarization_data(raw: dict, index: int) -> list[dict]:
    entries = []

    context = build_context(raw.get("document", ""))

    for label, key in [(0, "right_summary"), (1, "hallucinated_summary")]:
        response = clean_text(raw.get(key, ""))

        if not response:
            continue

        entries.append({
            "id": make_id("summarization", index, label),
            "source": "summarization",
            "context": context,
            "response": response,
            "label": label,
            "spans": [],
            "meta": {
                "context_len": len(context),
                "response_len": len(response)
            }
        })

    return entries



def load_json_or_jsonl(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("["):
        return json.loads(content)
    else:
        return [json.loads(line) for line in content.splitlines() if line.strip()]



# PIPELINE


PARSERS = {
    "dialogue": parse_dialogue_data,
    "general": parse_general_data,
    "qa": parse_qa_data,
    "summarization": parse_summarization_data,
}


def process_dataset(name: str, filepath: str) -> list[dict]:
    print(f"\n {name}")

    raw_records = load_json_or_jsonl(filepath)
    parser = PARSERS[name]

    unified = []
    for i, record in enumerate(raw_records):
        try:
            unified.extend(parser(record, i))
        except Exception as e:
            print(f"Erreur {i}: {e}")

    return unified



# SPLIT STRATIFIÉ (label + source)


def stratified_split(dataset, train_ratio=0.7, val_ratio=0.15):
    random.seed(42)

    groups = {}
    for d in dataset:
        key = (d["label"], d["source"])
        groups.setdefault(key, []).append(d)

    train, val, test = [], [], []

    for group in groups.values():
        random.shuffle(group)
        n = len(group)

        t = int(n * train_ratio)
        v = int(n * val_ratio)

        train.extend(group[:t])
        val.extend(group[t:t+v])
        test.extend(group[t+v:])

    return train, val, test


# MAIN


if __name__ == "__main__":

    DATA_FILES = {
        "dialogue": "data/dialogue_data.json",
        "general": "data/general_data.json",
        "qa": "data/qa_data.json",
        "summarization": "data/summarization_data.json",
    }

    all_entries = []

    for name, path in DATA_FILES.items():
        all_entries.extend(process_dataset(name, path))

    print(f"\nTotal: {len(all_entries)}")

    # Split
    train, val, test = stratified_split(all_entries)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Save
    os.makedirs("data/unified", exist_ok=True)

    def save(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    save(train, "data/unified/train.jsonl")
    save(val, "data/unified/val.jsonl")
    save(test, "data/unified/test.jsonl")

    print("\n Preprocessing terminé")