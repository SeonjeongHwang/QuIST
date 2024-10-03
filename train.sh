mkdir output

EPOCH=5
BATCH_SIZE=8
LR=1e-5
WARMUP=1000
EXP_DIR=output/QTC

#python run_QTC.py --do_train \
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