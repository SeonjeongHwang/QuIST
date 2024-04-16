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
from datasets import load_dataset
from transformers import MT5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, default_data_collator, EarlyStoppingCallback
from transformers import AutoModelForSeq2SeqLM
FeatInst = collections.namedtuple('FeatInst', 'unique_id input_ids attention_mask')

args = None
LANGCODE = {"en": "english", "bn": "bengali", "id": "indonesian", "sw": "swahili", "te": "telugu", "ko": "korean", "fi": "finnish"}

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
    parser.add_argument("--prompt_seed", type=int, default=None)
    parser.add_argument("--prompt_length", type=int, default=None)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_inference", action="store_true")
    parser.add_argument("--inference_dataset", type=str, choices=["tydiqa", "xquad", "squad", "None"], default="None")
    parser.add_argument("--inference_lang", type=str, default="")

    parser.add_argument("--do_generate", action="store_true")
    parser.add_argument("--ae_file", type=str, default=None)
    parser.add_argument("--cls_result_file", type=str, default=None)
    parser.add_argument("--generated_file", type=str, default=None)
    parser.add_argument("--generation_lang", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--exp_tag", type=str, default="test")
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--model_name", type=str, default="google/mt5-large")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_question_length", type=int, default=64)
    parser.add_argument("--frozen_list", type=str, default="", help="emb,dec_attn,dec_crossattn,dec_ffn,dec_final,lm_head")
    parser.add_argument("--example_origin", choices=["translated", "human"], default="human")
    parser.add_argument("--human_prompt_config", type=str, default=None)
    parser.add_argument("--example_type", type=str, default="specific", help="specific|all")
    parser.add_argument("--cls_result_dir", type=str, default=None)

    args = parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_samples(name, split):
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
                samples.append({"id": sample["id"], "context": context, "question": question, "answer": answer})
    elif args.inference_dataset == "xquad":
        assert lang in ["de", "zh", "hi"]
        s = load_dataset("xquad", f"xquad.{lang}", split="validation")
        for sample in s:
            context = sample["context"]
            question = sample["question"]
            answer = sample["answers"]["text"][0]
            samples.append({"id": sample["id"], "context": context, "question": question, "answer": answer})

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

    elif name == "squad":
        with open(f"data/SQuADv1.1_QT/{split}.json", "r") as fin:
            data_list = json.load(fin)["data"]
        for data in data_list:
            title = data["title"]
            paras = data["paragraphs"]
            for para in paras:
                context = f"[{title}] {para['context']}"
                for qa in para["qas"]:
                    question = qa["question"]
                    answer = qa["answers"][0]["text"]
                    answer_type = qa["answers"][0]["label"]
                    samples.append({"id": qa["id"], "context": context, "question": question, "answer": answer, "answer_type": answer_type})
    return samples

class Dataset(Dataset):
    def __init__(self, lang_samples, tokenizer, mode):
        super().__init__()
        self.tokenizer = tokenizer
        self.mode = mode
        is_inference = mode == "inference"

        print(f"Generate features...")
        self.features = []
        self.unique_id_to_gold = {}
        for lang, samples in lang_samples:
            if args.do_generate:
                self.partial_features, self.unique_id_to_gold = self.get_qg_syn_features(lang, self.unique_id_to_gold)
            else:
                self.partial_features, self.unique_id_to_gold = self.get_qg_features(lang, samples, self.unique_id_to_gold, is_inference)
            self.features += self.partial_features
        print("total feature num:", len(self.features))

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
    
    def get_qg_features(self, lang, samples, unique_id_to_gold, is_inference=False):
        label_to_examples = dict()
        if args.example_origin == "translated":
            if args.example_type == "specific":
                with open(f"data/interrogative_QT/seed{args.prompt_seed}.num{args.prompt_length}/{lang}.seed{args.prompt_seed}.num{args.prompt_length}.json", "r") as fin:
                    label_to_examples = json.load(fin)
            elif args.example_type == "all":
                with open(f"data/interrogative_QT/alltypes.seed{args.prompt_seed}/alltypes.{lang}.json", "r") as fin:
                    alltype_examples = json.load(fin)
        else: ### exemplars from human annotations (QA training datasets)
            if args.example_type == "specific":
                with open(f"data/prompts/{args.human_prompt_config}/{lang}.{args.human_prompt_config}.json", "r") as fin:
                    label_to_examples = json.load(fin)
            elif args.example_type == "all":
                with open(f"data/prompts_alltype/{lang}.seed{args.human_prompt_config}.json", "r") as fin:
                    alltype_examples = json.load(fin)

        id2predInterro = None
        if is_inference:
            id2predInterro = dict()
            cls_result_file = os.path.join(args.cls_result_dir, f"inference-{args.inference_dataset}.{lang}.json")
            with open(cls_result_file, "r") as fin:
                results = json.load(fin)["results"]

            for _id, values in results.items():
                id2predInterro[_id] = values["pred"]

        unique_id = len(unique_id_to_gold.keys())
        features = []
        passed = 0
        for sample in tqdm.tqdm(samples):
            _id = sample["id"]
            context = sample["context"]
            question = sample["question"]
            answer = sample["answer"]

            if is_inference:
                if args.example_type == "specific":
                    examples = label_to_examples[id2predInterro[_id]]
                else:
                    examples = alltype_examples
            else:
                if args.example_type == "specific":
                    answer_type = sample["answer_type"]
                    if answer_type == "NONE":
                        passed += 1
                        continue
                    examples = label_to_examples[answer_type]
                elif args.example_type == "all":
                    examples = alltype_examples

            example_sequence = f"Examples: {' '.join(examples)} "
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

        print(f"Passed: {passed}/{len(samples)}")
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
    pbar = tqdm.tqdm(dataloader, total=len(dataloader), desc=f"Inference")

    for batch in pbar:
        outputs = model.generate(input_ids=batch.input_ids.to(device),
                                 attention_mask=batch.attention_mask.to(device),
                                 num_beams=4,
                                 max_length=args.max_question_length,
                                 early_stopping=True)
        pred_sents = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for pred in pred_sents:
            predictions.append(pred)

        for unique_id in batch.unique_id:
            unique_id = unique_id.item()
            id_list.append(dataset.unique_id_to_gold[unique_id]["id"])
            references.append(dataset.unique_id_to_gold[unique_id]["question"])

    if args.do_generate:
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

    if not args.do_generate:
        os.makedirs(args.output_dir, exist_ok=True)
        model_dir = os.path.join(args.output_dir, args.exp_tag)
        os.makedirs(model_dir, exist_ok=True)

    if args.do_generate and args.checkpoint is None:
        print("Use median in best")
        with open("outputs/results/median_in_best.json", "r") as fin:
            length_seed = json.load(fin)[args.generation_lang]
        prompt_length = length_seed[0]
        prompt_seed = length_seed[1]
        print(f"PROMPT| length: {prompt_length} / seed: {prompt_seed}")

        model_dir = f"outputs/QG/pSeed.{prompt_seed}.pLength{prompt_length}-10.48.5e-5.1000"

    config = os.path.join(model_dir, "config.json")
    if not args.do_train:
        with open(config, "r") as fin:
            arg_dict = json.load(fin)
            args.seed = arg_dict["seed"]
            args.model_name = arg_dict["model_name"]
            args.max_seq_length = arg_dict["max_seq_length"]
            args.max_question_length = arg_dict["max_question_length"]
            args.frozen_list = arg_dict["frozen_list"]
            args.example_type = arg_dict["example_type"]
            args.prompt_seed = arg_dict["prompt_seed"]
            args.prompt_length = arg_dict["prompt_length"]
            args.example_type = arg_dict["example_type"]
    print(args)

    if args.do_train and not args.use_checkpoint:
        config = os.path.join(model_dir, "config.json")
        with open(config, "w") as fout:
            json.dump(vars(args), fout, indent=1)

    seed_everything(args.seed)
    device = torch.device("cuda")

    tokenizer = MT5Tokenizer.from_pretrained(args.model_name)
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

    if args.do_train:
        ### Load Dataset
        task_samples = {"train": [], "valid": []}
        train_samples = get_samples("squad", "train")
        valid_samples = get_samples("squad", "dev")
        task_samples["train"].append(("en", train_samples))
        task_samples["valid"].append(("en", valid_samples))

        print(f"#train samples:{len(train_samples)}")
        print(f"#valid samples:{len(valid_samples)}")

        print("Generate train features...")
        train_dataset = Dataset(task_samples["train"], tokenizer, "train")

        print("Generate valid features...")
        valid_dataset = Dataset(task_samples["valid"], tokenizer, "valid")


        training_args = Seq2SeqTrainingArguments(
            output_dir=model_dir,
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

        trainer.train(resume_from_checkpoint=args.use_checkpoint)

    if args.do_test:
        checkpoint_dir = os.path.join(model_dir, f"checkpoint-{args.checkpoint}")        
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        model = model.to(device)

        ### Load Dataset
        task_samples = []
        task_samples.append(("en", get_samples("squad", "test")))

        print("Generate test features...")
        test_dataset = Dataset(task_samples, tokenizer, "test")

        result_file = os.path.join(checkpoint_dir, f"test_en.json")
        inference(model, tokenizer, test_dataset, result_file)

    if args.do_inference:
        assert args.cls_result_dir is not None
        assert args.inference_lang != ""

        if args.checkpoint is None:
            checkpoint = [x for x in os.listdir(model_dir) if x.startswith("checkpoint")][0]
            checkpoint_dir = os.path.join(model_dir, checkpoint)
        else:
            checkpoint_dir = os.path.join(model_dir, f"checkpoint-{args.checkpoint}")      
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        model = model.to(device)

        ### Load Dataset
        for lang in args.inference_lang.split(","):
            task_samples = []
            print(lang)
            task_samples.append((lang, get_samples(lang, "test")))

            print("Generate test features...")
            test_dataset = Dataset(task_samples, tokenizer, "inference")

            result_file = os.path.join(checkpoint_dir, f"inference_human_{args.inference_dataset}.{lang}.{args.human_prompt_config}.json")
            inference(model, tokenizer, test_dataset, result_file)

    if args.do_generate:
        assert args.cls_result_file
        assert args.ae_file
        assert args.generated_file

        if args.checkpoint is None:
            checkpoint = [x for x in os.listdir(model_dir) if x.startswith("checkpoint")][0]
            checkpoint_dir = os.path.join(model_dir, checkpoint)
        else:
            checkpoint_dir = os.path.join(model_dir, f"checkpoint-{args.checkpoint}")        
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
        model = model.to(device)

        task_samples = []
        task_samples.append((args.generation_lang, None))

        print("Generate test features...")
        test_dataset = Dataset(task_samples, tokenizer, "inference")

        inference(model, tokenizer, test_dataset, args.generated_file)

