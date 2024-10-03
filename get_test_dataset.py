import json
from datasets import load_dataset

LANGCODE = {"bn": "bengali", "id": "indonesian", "sw": "swahili", "te": "telugu", "ko": "korean", "fi": "finnish"}
for lang in ["bn", "id", "sw", "te", "ko", "fi"]:
    print(lang)
    s = load_dataset("tydiqa", "secondary_task", split="validation")
    lang_name = LANGCODE[lang]

    data_list = []
    for sample in s:
        if sample["id"].startswith(lang_name):
            context = f"[{sample['title']}] {sample['context']}"
            question = sample["question"]
            answer = sample["answers"]["text"][0]
            data_list.append({"id": sample["id"], "context": context, "question": question, "answer": answer})

    with open(f"data/target/{lang}.test.json", "w") as fout:
        json.dump(data_list, fout, indent=1)

for lang in ["de", "zh", "hi"]:
    print(lang)
    s = load_dataset("xquad", f"xquad.{lang}", split="validation")

    data_list = []
    for sample in s:
        context = sample["context"]
        question = sample["question"]
        answer = sample["answers"]["text"][0]
        data_list.append({"id": sample["id"], "context": context, "question": question, "answer": answer})

    with open(f"data/target/{lang}.test.json", "w") as fout:
        json.dump(data_list, fout, indent=1)