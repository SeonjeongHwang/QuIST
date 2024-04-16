import os
import json
import tqdm
import random
import numpy as np
import argparse
import collections
import nltk

import evaluate

import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from datasets import load_dataset
from transformers import  AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, default_data_collator, EarlyStoppingCallback
FeatInst = collections.namedtuple('FeatInst', 'unique_id input_ids attention_mask token_type_ids labels')

args = None
LANGCODE = {"en": "english", "bn": "bengali", "id": "indonesian", "sw": "swahili", "te": "telugu", "ko": "korean", "fi": "finnish", "de": "german", "zh": "chinese", "hi": "hindi"}

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
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_inference", action="store_true")

    parser.add_argument("--do_generate", action="store_true")
    parser.add_argument("--generation_input_file", type=str, default=None)
    parser.add_argument("--generated_file", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--exp_tag", type=str, default="test")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--inference_dataset", type=str, choices=["tydiqa", "xquad", "squad"])
    parser.add_argument("--inference_lang", type=str, default=None)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--label_balance", action="store_true")
    parser.add_argument("--use_answer_context", action="store_true")

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
    def __init__(self, lang, tokenizer, mode):
        assert lang in LANGCODE.keys()
        super().__init__()
        self.tokenizer = tokenizer
        self.mode = mode

        print(f"Generate features...")
        if args.do_generate:
            self.features, self.unique_id_to_gold = self.get_syn_features()

        else:
            samples = []
            if args.inference_dataset == "tydiqa":
                assert lang in ["bn", "id", "sw", "te", "ko", "fi"]
                s = load_dataset("tydiqa", "secondary_task", split="validation")
                lang_name = LANGCODE[lang]
                for sample in s:
                    if sample["id"].startswith(lang_name):
                        context = f"[{sample['title']}] {sample['context']}"
                        question = sample["question"]
                        answer = sample["answers"]["text"][0]
                        answer_start = sample["answers"]["answer_start"][0] + len(f"[{sample['title']}] ")
                        samples.append({"id": sample["id"], "context": context, "question": question, "answer": answer, "answer_start": answer_start}) 

            elif args.inference_dataset == "xquad":
                assert lang in ["de", "zh", "hi"]
                s = load_dataset("xquad", f"xquad.{lang}", split="validation")
                for sample in s:
                    context = sample["context"]
                    question = sample["question"]
                    answer = sample["answers"]["text"][0]
                    answer_start = sample["answers"]["answer_start"][0]
                    samples.append({"id": sample["id"], "context": context, "question": question, "answer": answer, "answer_start": answer_start})

            elif args.inference_dataset == "squad":
                assert lang in ["en"]
                with open(f"data/SQuADv1.1_newsplits/test.json", "r") as fin:
                    data_list = json.load(fin)["data"]
                for data in data_list:
                    title = data["title"]
                    paras = data["paragraphs"]
                    for para in paras:
                        context = f"[{title}] {para['context']}"
                        for qa in para["qas"]:
                            question = qa["question"]
                            answer = qa["answers"][0]["text"]
                            samples.append({"id": qa["id"], "context": context, "question": question, "answer": answer})
                            
            elif lang == "en":
                with open(f"data/SQuADv1.1_QT/{mode}.json", "r") as fin:
                    data_list = json.load(fin)["data"]
                for data in data_list:
                    title = data["title"]
                    paras = data["paragraphs"]
                    for para in paras:
                        context = f"[{title}] {para['context']}"
                        for qa in para["qas"]:
                            question = qa["question"]
                            answer = qa["answers"][0]["text"]
                            answer_start = qa["answers"][0]["answer_start"] + len(f"[{sample['title']}] ")
                            label = qa["answers"][0]["label"]
                            samples.append({"id": qa["id"], "context": context, "question": question, "answer": answer, "answer_start": answer_start, "label": label})

            self.features, self.unique_id_to_gold = self.get_features(samples, mode)
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
    
    def get_features(self, samples, mode):
        unique_id = 0
        unique_id_to_gold = dict()
        features = []
        label_to_features = dict([(i, []) for i, label in enumerate(LABEL_LIST)])

        passed = 0
        for sample in tqdm.tqdm(samples):
            context = sample["context"]
            question = sample["question"]
            answer = sample["answer"]

            if args.use_answer_context:
                answer_start = sample["answer_start"]
                answer_end = answer_start + len(answer) - 1

                context_sentences = []
                for sent in nltk.sent_tokenize(context):
                    sent_start = context.find(sent)
                    sent_end = sent_start + len(sent) - 1
                    if not (answer_end < sent_start or sent_end < answer_start):
                        context_sentences.append(sent)
                context = " ".join(context_sentences)

            if mode != "inference":
                if sample["label"] == "NONE":
                    passed += 1
                    continue
                label = LABEL_LIST.index(sample["label"])
            else:
                label = -1
            
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
            if mode == "train":
                label_to_features[label].append(feature)
            unique_id_to_gold[unique_id] = {"id": sample["id"], 
                                            "question": question,
                                            "answer": answer}
            unique_id += 1

        print(f"Passed: {passed}/{len(samples)}")
        if args.label_balance and mode == "train":
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

        return features, unique_id_to_gold
    
    def get_syn_features(self):
        unique_id = 0
        unique_id_to_gold = dict()
        features = []

        with open(args.generation_input_file, "r") as fin:
            data_dict = json.load(fin)

        for _id, data in tqdm.tqdm(data_dict.items(), total=len(data_dict.keys())):
            document = data["context"]
            for aid, answer in enumerate(data["pred"]):
                answer_text = answer["text"]
                context = document[:]
                
                if args.use_answer_context:
                    answer_start = answer["char_start"]
                    answer_end = answer["char_end"]

                    context_sentences = []
                    for sent in nltk.sent_tokenize(context):
                        sent_start = context.find(sent)
                        sent_end = sent_start + len(sent) - 1
                        if not (answer_end < sent_start or sent_end < answer_start):
                            context_sentences.append(sent)
                    context = " ".join(context_sentences)

                label = -1

                inputs = self.tokenizer(answer_text, context, max_length=args.max_seq_length, truncation="only_second", padding='max_length')
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
                unique_id_to_gold[unique_id] = {"id": f"{_id}-{aid}",
                                                "question": "Empty",
                                                "answer": answer_text}
                unique_id += 1         

        return features, unique_id_to_gold

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
    for unique_id, pred, ref in zip(unique_ids, preds, golds):
        _id = dataset.unique_id_to_gold[unique_id]["id"]
        question = dataset.unique_id_to_gold[unique_id]["question"]
        answer = dataset.unique_id_to_gold[unique_id]["answer"]

        pred_interro = label2interro[pred]
        ref_interro = label2interro[ref]

        results[_id]= {"question": question,
                       "answer": answer,
                       "gold": ref_interro,
                       "pred": pred_interro}
        
    with open(result_file, "w", encoding="UTF-8") as fout:
        json.dump({"scores": scores, "results": results}, fout, indent=1, ensure_ascii=False)
            
if __name__ == "__main__":
    parse_argument()

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, args.exp_tag)
    os.makedirs(model_dir, exist_ok=True)
    

    config = os.path.join(model_dir, "config.json")
    if args.use_checkpoint or args.checkpoint:
        with open(config, "r") as fin:
            arg_dict = json.load(fin)
            args.seed = arg_dict["seed"]
            args.model_name = arg_dict["model_name"]
            args.max_seq_length = arg_dict["max_seq_length"]
            args.use_answer_context = arg_dict["use_answer_context"]
    print(args)

    if args.do_train:
        config = os.path.join(model_dir, "config.json")
        with open(config, "w") as fout:
            json.dump(vars(args), fout, indent=1)

    seed_everything(args.seed)
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABEL_LIST))
    print(model) 

    if args.do_train:
        ### Load Dataset
        print("Generate train features...")
        train_dataset = Dataset("en", tokenizer, "train")

        print("Generate valid features...")
        valid_dataset = Dataset("en", tokenizer, "dev")

        
        training_args = TrainingArguments(
            output_dir=model_dir,
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

        trainer.train(resume_from_checkpoint=args.use_checkpoint)

    if args.do_test:
        ### Load Dataset
        print("Generate valid features...")
        test_dataset = Dataset("en", tokenizer, "test")

        assert args.checkpoint is not None
        checkpoint_dir = os.path.join(model_dir, f"checkpoint-{args.checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        model = model.to(device)

        result_file = os.path.join(checkpoint_dir, f"test-en.json")
        inference(model, test_dataset, device, result_file)

    if args.do_inference:
        assert args.inference_lang is not None
        inference_langs = args.inference_lang.split(",")
        for lang in inference_langs:
            ### Load Dataset
            print("Lang:", lang)
            print("Generate Inference features...")
            test_dataset = Dataset(lang, tokenizer, "inference")

            assert args.checkpoint is not None
            checkpoint_dir = os.path.join(model_dir, f"checkpoint-{args.checkpoint}")
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
            model = model.to(device)

            result_file = os.path.join(checkpoint_dir, f"inference-{args.inference_dataset}.{lang}.json")
            inference(model, test_dataset, device, result_file)

    if args.do_generate:
        test_dataset = Dataset(None, tokenizer, "generation")

        assert args.checkpoint is not None
        checkpoint_dir = os.path.join(model_dir, f"checkpoint-{args.checkpoint}")
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
        model = model.to(device)

        inference(model, test_dataset, device, result_file=args.generated_file)