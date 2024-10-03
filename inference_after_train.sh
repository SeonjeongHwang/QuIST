### Download target language datasets
### Data is downloaded to data/target/
#python get_test_dataset.py

LANG=ko  ### ["bn", "de", "fi", "hi", "id", "ko", "te", "sw", "zh"]
python run_QTC.py --do_inference \
                  --inference_lang $LANG \
                  --exp_dir output/QTC

python run_QG.py --do_inference \
                 --inference_lang $LANG \
                 --exp_dir output/QG

