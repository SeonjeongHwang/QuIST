import os
import json
import tqdm
import random
import numpy as np
import argparse
import collections

import nltk
from rouge import Rouge

import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from transformers import MT5Tokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator, EarlyStoppingCallback
FeatInst = collections.namedtuple('FeatInst', 'unique_id input_ids attention_mask')

args = None

def parse_argument():
    global args, lang_dict, task_dict
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--valid_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--early_stop", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--inference_lang", choices=["bn", "de", "fi", "hi", "id", "ko", "te", "sw", "zh"])

    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="google/mt5-large")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_question_length", type=int, default=64)
    parser.add_argument("--frozen_list", type=str, default="", help="emb,dec_attn,dec_crossattn,dec_ffn,dec_final,lm_head")

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
        self.features, self.unique_id_to_gold = self.get_features()

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

    def collate_fn(self, batch):
        for i, feature in enumerate(batch):
            batch[i] = FeatInst(unique_id=np.asarray(feature["unique_id"]),
                                input_ids=np.asarray(feature["input_ids"]),
                                attention_mask=np.asarray(feature["attention_mask"]))
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
                        question = qa["question"]
                        answer = qa["answers"][0]["text"]
                        question_type = qa["answers"][0]["label"]
                        if question_type == "NONE":
                            continue
                        data_list.append({"id": qa["id"], "context": context, "question": question, "answer": answer, "question_type": question_type})
        else:
            assert self.split == "inference"
            data_list = json.load(open(f"data/target/{self.lang}.test.json"))

        return data_list
    
    def get_features(self):
        data_list = self.get_data_list()
        label2exemplars = json.load(open(f"data/exemplars/{self.lang}.json"))

        id2predInterro = None
        if self.split == "inference":
            id2predInterro = dict()
            with open(f"evaluation_results/qtc.evaluation_{self.lang}.json", "r") as fin:
                results = json.load(fin)["results"]

            for _id, values in results.items():
                id2predInterro[_id] = values["pred"]

        unique_id = 0
        unique_id_to_gold = dict()
        features = []
        for data in tqdm.tqdm(data_list):
            _id = data["id"]
            context = data["context"]
            question = data["question"]
            answer = data["answer"]

            if self.split == "inference":
                question_type = id2predInterro[_id]
            else:
                question_type = data["question_type"]
            
            exemplars = label2exemplars[question_type]

            example_sequence = f"Examples: {' '.join(exemplars)} "
            input_sequence = example_sequence + f"Answer: {answer}. Context: {context}"
            input_ids = self.tokenizer.encode(input_sequence)

            if len(input_ids) > args.max_seq_length:
                input_ids = input_ids[:args.max_seq_length]
            
            attention_mask = [1]*len(input_ids)
            while len(input_ids) < args.max_seq_length:
                input_ids.append(self.tokenizer.pad_token_id)
                attention_mask.append(0)

            label = self.tokenizer.encode(question)
            if len(label) > args.max_question_length:
                label = label[:args.max_question_length]
            
            while len(label) < args.max_question_length:
                label.append(-100)

            feature = {"unique_id": unique_id,
                       "input_ids": input_ids,
                       "attention_mask": attention_mask,
                       "labels": label}
            unique_id_to_gold[unique_id] = {"id": _id,
                                            "question": question}
            features.append(feature)
            unique_id += 1

        return features, unique_id_to_gold
    
def get_scores(predictions, references):
    preds = []
    golds = []
    m_score = 0.0
    rl_score = 0.0
    rouge = Rouge()
    num = 0

    for pred, gold in zip(predictions, references):
        if pred.strip() == "":
            pred = "None"
        rl_score += rouge.get_scores([pred], [gold])[0]['rouge-l']['f']
        pred = nltk.word_tokenize(pred)
        gold = nltk.word_tokenize(gold)
        preds.append(pred)
        golds.append([gold])
        num += 1
        m_score += nltk.translate.meteor_score.meteor_score([gold], pred)
    bleu_score = nltk.translate.bleu_score.corpus_bleu(golds, preds) * 100
    m_score /= num
    m_score *= 100
    rl_score /= num
    rl_score *= 100

    return bleu_score, m_score, rl_score

def inference(model, tokenizer, dataset, result_file):
    model.eval()
    id_list = []
    predictions = []
    references = []
    
    dataloader = DataLoader(dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    pbar = tqdm.tqdm(dataloader, total=len(dataloader), desc=f"Evaluation")

    for batch in pbar:
        outputs = model.generate(input_ids=batch.input_ids.to(device),
                                 attention_mask=batch.attention_mask.to(device),
                                 num_beams=4,
                                 max_length=args.max_question_length,
                                 early_stopping=True)
        pred_sents = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for pred in pred_sents:
            predictions.append(pred.replace("▁<extra_id_0>", ""))

        for unique_id in batch.unique_id:
            unique_id = unique_id.item()
            id_list.append(dataset.unique_id_to_gold[unique_id]["id"])
            references.append(dataset.unique_id_to_gold[unique_id]["question"])

    if args.do_inference:
        results = dict()
        for _id, pred in zip(id_list, predictions):
            results[_id] = pred

        with open(result_file, "w", encoding="UTF-8") as fout:
            json.dump({"results": results}, fout, indent=1, ensure_ascii=False)

        return True


    print("Evaluation Result")
    bleu, meteor, rougel = get_scores(predictions, references)

    if result_file:
        results = dict()
        for _id, gold, pred in zip(id_list, references, predictions):
            results[_id] = {"gold": gold,
                            "pred": pred}
        with open(result_file, "w", encoding="UTF-8") as fout:
            json.dump({"Score": {"BLEU": bleu, "METEOR": meteor, "ROUGE-L": rougel}, "results": results}, fout, indent=1, ensure_ascii=False)

    return (bleu+meteor+rougel)/3

def compute_metrics(EvalPred):
    total_preds, total_labels = EvalPred
    if isinstance(total_preds, tuple):
        total_preds = total_preds[0]

    predictions, references = [], []
    for pred, label in zip(total_preds, total_labels):
        predictions.append(pred.tolist())
        references.append(label.tolist())
    predictions = np.array(predictions)
    references = np.array(references)

    scores = dict()    
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    references = np.where(references != -100, references, tokenizer.pad_token_id)
    references = tokenizer.batch_decode(references, skip_special_tokens=True)

    bleu, meteor, rougel = get_scores(predictions, references)
    avg_score = (bleu+meteor+rougel)/3

    scores[f"bleu"] = round(bleu, 2)
    scores[f"meteor"] = round(meteor, 2)
    scores[f"rougel"] = round(rougel, 2)
    scores[f"avg_score"] = round(avg_score, 2)

    return scores
            
if __name__ == "__main__":
    parse_argument()

    if args.exp_dir:
        os.makedirs(args.exp_dir, exist_ok=True)
        config = os.path.join(args.exp_dir, "config.json")

    if args.do_train:
        with open(config, "w") as fout:
            json.dump(vars(args), fout, indent=1)
    elif args.exp_dir:
        config_dict = json.load(open(config))
        args.max_seq_length = config_dict["max_seq_length"]
        args.max_question_length = config_dict["max_question_length"]

    seed_everything(args.seed)
    device = torch.device("cuda")
    tokenizer = MT5Tokenizer.from_pretrained(args.model_name)

    if args.do_train:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

        # Freeze components
        for name, param in model.named_parameters():
            if "emb" in args.frozen_list:
                if "shared" in name or "embed_tokens" in name:
                    param.requires_grad = False
                    
            if "enc_attn" in args.frozen_list:
                if "encoder" in name and "layer.0" in name:
                    param.requires_grad = False
            
            if "enc_ffn" in args.frozen_list:
                if "encoder" in name and "layer.1" in name:
                    param.requires_grad = False

            if "enc_final" in args.frozen_list:
                if "encoder" in name and "final_layer" in name:
                    param.requires_grad = False

            if "dec_attn" in args.frozen_list:
                if "decoder" in name and "layer.0" in name:
                    param.requires_grad = False
            
            if "dec_crossattn" in args.frozen_list:
                if "decoder" in name and "layer.1" in name:
                    param.requires_grad = False

            if "dec_ffn" in args.frozen_list:
                if "decoder" in name and "layer.2" in name:
                    param.requires_grad = False

            if "dec_final" in args.frozen_list:
                if "decoder" in name and "final_layer" in name:
                    param.requires_grad = False

            if "lm_head" in args.frozen_list:
                if "lm_head" in name:
                    param.requires_grad = False

        ### Load Dataset
        print("Generate train features...")
        train_dataset = Dataset(tokenizer, "en", "train")
        print("Generate valid features...")
        valid_dataset = Dataset(tokenizer, "en", "dev")

        training_args = Seq2SeqTrainingArguments(
            output_dir=args.exp_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.valid_batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            seed=args.seed,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model=f"eval_avg_score",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=args.max_question_length,
            generation_num_beams=4,
            report_to="none"
        )

        model = model.to(device)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        trainer.save_model(f"{args.exp_dir}/best")

    os.makedirs("evaluation_results", exist_ok=True)

    if args.do_eval:
        if args.exp_dir:
            checkpoint_dir = f"{args.exp_dir}/best"     
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        model = model.to(device)

        ### Load Dataset
        print("Generate test features...")
        test_dataset = Dataset(tokenizer, "en", "dev")

        result_file = os.path.join("evaluation_results", f"qg.evaluation_en.json")
        inference(model, tokenizer, test_dataset, result_file)

    if args.do_inference:
        if args.exp_dir:
            checkpoint_dir = f"{args.exp_dir}/best"     
            model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
        model = model.to(device)

        print("Generate test features...")
        test_dataset = Dataset(tokenizer, args.inference_lang, "inference")

        result_file = os.path.join("evaluation_results", f"qg.evaluation_{args.inference_lang}.json")
        inference(model, tokenizer, test_dataset, result_file)

