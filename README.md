### Cross-lingual Transfer for Automatic Question Generation by Learning Interrogative Structures in Target Languages (EMNLP 2024)
---
This repository contains code for
1. Training QTC and QG models
2. Inference using trained model checkpoints
3. Inference using our released checkpoints

# Environment
```
torch
transformers
nltk
rouge
evaluate
sentencepiece
```

# Data
+ English QA dataset: [SQuADv1.1](https://huggingface.co/datasets/rajpurkar/squad) dataset labeled with question type is in `data/SQuADv1.1_QT`.
+ Question Exemplars: English and target language question exemplars are in `data/exemplars`. We provide question exemplars for nine target languages: Bengali, Chinese, Finnish, German, Hindi, Indonesian, Korean, Swahili, Telugu.
+ Target langauge datasets: You can download the test datasets for the target languages from [XQuAD](https://huggingface.co/datasets/google/xquad) and [TyDiQA](google-research-datasets/tydiqa) by running the provided code. The datasets will be saved in `data/target`.
```
python get_test_dataset.py
```
+ You can also test on additional languages by using your own question exemplars and datasets containing context-answer pairs. Please ensure that your data follows the format of the provided files (use "-" as a placeholder for the question when testing new languages).

# Inference
We provide checkpoints for our QTC and QG models. The QG model was trained using English exemplars, with 15 questions for each question type. The inference code below downloads the checkpoints and generates questions for the target language `LANG`. The model checkpoints can be found here: [QuIST/QTC](https://huggingface.co/seonjeongh/QuIST-QTC), [QuIST/QG-15](https://huggingface.co/seonjeongh/QuIST-QG-15).
```
sh inference_only.sh
```
or
```
LANG=ko  # ["bn", "de", "fi", "hi", "id", "ko", "te", "sw", "zh"]
python run_QTC.py --do_inference \
                  --inference_lang $LANG \
                  --model_name seonjeongh/QuIST-QTC

python run_QG.py --do_inference \
                 --inference_lang $LANG \
                 --model_name seonjeongh/QuIST-QG-15
```

# Training and Inference
You can also train and test your model on the new English QA dataset.
### Training
```
sh train.sh
```
or
```
mkdir output

EPOCH=5
BATCH_SIZE=8
LR=1e-5
WARMUP=1000
EXP_DIR=output/QTC

python run_QTC.py --do_train \
                  --epochs $EPOCH \
                  --batch_size $BATCH_SIZE \
                  --valid_batch_size $BATCH_SIZE \
                  --learning_rate $LR \
                  --warmup_steps $WARMUP \
                  --exp_dir $EXP_DIR \
                  --label_balance

EPOCH=10
BATCH_SIZE=16
LR=5e-5
WARMUP=1000
EXP_DIR=output/QG

python run_QG.py --do_train \
                 --epochs $EPOCH \
                 --batch_size $BATCH_SIZE \
                 --valid_batch_size $BATCH_SIZE \
                 --learning_rate $LR \
                 --warmup_steps $WARMUP \
                 --exp_dir $EXP_DIR \
                 --frozen_list emb,dec_attn,dec_crossattn,dec_ffn,dec_final,lm_head
```
### Inference
```
sh inference_after_train.sh
```
or
```
LANG=ko  # ["bn", "de", "fi", "hi", "id", "ko", "te", "sw", "zh"]
python run_QTC.py --do_inference \
                  --inference_lang $LANG \
                  --exp_dir output/QTC

python run_QG.py --do_inference \
                 --inference_lang $LANG \
                 --exp_dir output/QG
```
