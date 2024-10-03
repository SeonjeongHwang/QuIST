import os
import json
import tqdm
import random
import numpy as np
import argparse
import collections

import evaluate

import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from transformers import  AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, default_data_collator, EarlyStoppingCallback
FeatInst = collections.namedtuple('FeatInst', 'unique_id input_ids attention_mask token_type_ids labels')

args = None

with open("data/interro_list.json", "r") as fin:
    LABEL_LIST = json.load(fin)
    
def parse_argument():
    global args, lang_dict, task_dict
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--early_stop", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--inference_lang", choices=["bn", "de", "fi", "hi", "id", "ko", "te", "sw", "zh"])

    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--label_balance", action="store_true")

    args = parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Dataset(Dataset):
    def __init__(self, tokenizer, lang, split):
        super().__init__()
        self.tokenizer = tokenizer
        self.lang = lang
        self.split = split

        print(f"Generate features...")
        self.features, self.unique_id_to_id = self.get_features()
        print("feature num:", len(self.features))

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

    def collate_fn(self, batch):
        for i, feature in enumerate(batch):
            batch[i] = FeatInst(unique_id=np.asarray(feature["unique_id"]),
                                input_ids=np.asarray(feature["input_ids"]),
                                attention_mask=np.asarray(feature["attention_mask"]),
                                token_type_ids=np.asarray(feature["token_type_ids"]),
                                labels=np.asarray(feature["labels"]))
        results = FeatInst(*(default_collate(samples) for samples in zip(*batch)))
        return results
    
    def get_data_list(self):
        data_list = []
        if self.lang == "en":
            with open(f"data/SQuADv1.1_QT/{self.split}.json", "r") as fin:
                samples = json.load(fin)["data"]
            for sample in samples:
                title = sample["title"]
                paras = sample["paragraphs"]
                for para in paras:
                    context = f"[{title}] {para['context']}"
                    for qa in para["qas"]:
                        answer = qa["answers"][0]["text"]
                        question_type = qa["answers"][0]["label"]
                        data_list.append({"id": qa["id"], "context": context, "answer": answer, "question_type": question_type})
        else:
            assert self.lang in ["bn", "de", "fi", "id", "ko", "te", "sw", "zh"]
            assert self.split == "inference"
            data_list = json.load(open(f"data/target/{self.lang}.test.json"))

        return data_list
    
    def get_features(self):
        data_list = self.get_data_list()

        unique_id = 0
        unique_id_to_id = dict()
        features = []
        label_to_features = dict([(i, []) for i, _ in enumerate(LABEL_LIST)])

        for data in tqdm.tqdm(data_list):
            context = data["context"]
            answer = data["answer"]

            if self.split == "inference":
                label = -1
            else:
                question_type = data["question_type"]
                if question_type == "NONE":
                    continue
                label = LABEL_LIST.index(question_type)
            
            inputs = self.tokenizer(answer, context, max_length=args.max_seq_length, truncation="only_second", padding='max_length')
            assert len(inputs["input_ids"]) == args.max_seq_length

            if "token_type_ids" in inputs.keys():
                feature = {"unique_id": unique_id,
                           "input_ids": inputs["input_ids"],
                           "attention_mask": inputs["attention_mask"],
                           "token_type_ids": inputs["token_type_ids"],
                           "labels": label}
            else:
                feature = {"unique_id": unique_id,
                           "input_ids": inputs["input_ids"],
                           "attention_mask": inputs["attention_mask"],
                           "labels": label}
            features.append(feature)
            if self.split == "train":
                label_to_features[label].append(feature)
            unique_id_to_id[unique_id] = data["id"]
            unique_id += 1

        if args.label_balance and self.split == "train":
            print("Original label distribution:", dict((k, len(v)) for k,v in label_to_features.items()))
            max_num = -1
            for v in label_to_features.values():
                if len(v) > max_num:
                    max_num = len(v)
            features = []
            for v in label_to_features.values():
                if len(v) == max_num:
                    features += v
                else:
                    aug_features = []
                    while len(aug_features) < max_num:
                        more = min(len(v), max_num-len(aug_features))
                        np.random.shuffle(v)
                        aug_features += v[:more]
                    assert len(aug_features) == max_num
                    features += aug_features            

        return features, unique_id_to_id

def compute_metrics(EvalPred):
    total_preds, references = EvalPred
    if isinstance(total_preds, tuple):
        total_preds = total_preds[0]
        print("tuple!!")

    predictions = np.argmax(np.array(total_preds), axis=-1)

    f1_metric = evaluate.load("f1")
    macro_f1 = f1_metric.compute(predictions=predictions, references=references, average="macro")["f1"]
    micro_f1 = f1_metric.compute(predictions=predictions, references=references, average="micro")["f1"]
    results = {"macro_f1": macro_f1, "micro_f1": micro_f1}

    return results

def inference(model, dataset, device, result_file):
    model.eval()

    unique_ids = []
    preds = []
    golds = []
    dataloader = DataLoader(dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    pbar = tqdm.tqdm(dataloader, total=len(dataloader))

    for batch in pbar:
        if "bert" in args.model_name:
            outputs = model(input_ids=batch.input_ids.to(device),
                            attention_mask=batch.attention_mask.to(device),
                            token_type_ids=batch.token_type_ids.to(device)).logits
        else:
            outputs = model(input_ids=batch.input_ids.to(device),
                            attention_mask=batch.attention_mask.to(device)).logits

        outputs = outputs.detach().cpu().numpy()
        preds += list(np.argmax(outputs, axis=-1))
        golds += list(batch.labels.detach().numpy())
        unique_ids += list(batch.unique_id.detach().numpy())

    label2interro = dict([(i, interro) for i, interro in enumerate(LABEL_LIST)])
    label2interro[-1] = "No-Label"

    predictions = []
    references = []
    for pred, gold in zip(preds, golds):
        if gold == -1:
            continue
        predictions.append(pred)
        references.append(gold)

    if len(references) > 0:
        f1_metric = evaluate.load("f1")
        macro_f1 = f1_metric.compute(predictions=predictions, references=references, average="macro")["f1"]
        micro_f1 = f1_metric.compute(predictions=predictions, references=references, average="micro")["f1"]
        scores = {"macro_f1": macro_f1, "micro_f1": micro_f1}
        print(scores)
    else:
        scores = None

    results = dict()
    for unique_id, pred, gold in zip(unique_ids, preds, golds):
        _id = dataset.unique_id_to_id[unique_id]

        pred_interro = label2interro[pred]
        gold_interro = label2interro[gold]

        results[_id]= {"gold": gold_interro,
                       "pred": pred_interro}
        
    with open(result_file, "w", encoding="UTF-8") as fout:
        json.dump({"scores": scores, "results": results}, fout, indent=1, ensure_ascii=False)
            
if __name__ == "__main__":
    parse_argument()

    if args.exp_dir:
        os.makedirs(args.exp_dir, exist_ok=True)
        config = os.path.join(args.exp_dir, "config.json")

    if args.do_train:
        config = os.path.join(args.exp_dir, "config.json")
        with open(config, "w") as fout:
            json.dump(vars(args), fout, indent=1)
    elif args.exp_dir:
        config_dict = json.load(open(config))
        args.max_seq_length = config_dict["max_seq_length"]

    seed_everything(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.do_train:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABEL_LIST))

        ### Load Dataset
        print("Generate train features...")
        train_dataset = Dataset(tokenizer, "en", "train")
        print("Generate valid features...")
        valid_dataset = Dataset(tokenizer, "en", "dev")

        training_args = TrainingArguments(
            output_dir=args.exp_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.valid_batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            save_strategy="steps",
            evaluation_strategy="steps",
            save_steps=5000,
            eval_steps=5000,
            seed=args.seed,
            save_total_limit=args.early_stop,
            load_best_model_at_end=True,
            metric_for_best_model=f"eval_macro_f1",
            greater_is_better=True,
            report_to="none"
        )

        if args.freeze_encoder:
            for param in model.bert.parameters():
                param.requires_grad = False

            unfrozen = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    unfrozen.append(name)

            print("Unfrozen:", unfrozen)

        model = model.to(device)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stop)]
        )

        trainer.train()
        trainer.save_model(f"{args.exp_dir}/best")

    os.makedirs("evaluation_results", exist_ok=True)

    if args.do_eval:
        ### Load Dataset
        print("Generate valid features...")
        test_dataset = Dataset(tokenizer, "en", "dev")

        if args.exp_dir:
            checkpoint_dir = f"{args.exp_dir}/best"     
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        model = model.to(device)

        result_file = os.path.join("evaluation_results", f"qtc.evaluation_en.json")
        inference(model, test_dataset, device, result_file)

    if args.do_inference:
        test_dataset = Dataset(tokenizer, args.inference_lang, "inference")

        if args.exp_dir:
            checkpoint_dir = f"{args.exp_dir}/best"     
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        model = model.to(device)

        result_file = os.path.join("evaluation_results", f"qtc.evaluation_{args.inference_lang}.json")
        inference(model, test_dataset, device, result_file)