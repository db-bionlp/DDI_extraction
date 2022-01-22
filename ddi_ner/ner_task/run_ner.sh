#############JNLPBA
python3 main_run.py \
  --filename ../ner_dataset/JNLPBA \
  --data_dir ../ner_dataset/JNLPBA  \
  --label_file ../ner_dataset/labels.txt  \
  --model_dir ../saved_model/JNLPBA/JNLPBA_scibert_01 \
  --model_type scibert  \
  --model only_bert  \
  --per_gpu_train_batch_size=8  \
  --per_gpu_eval_batch_size=32  \
  --max_steps=-1  \
  --num_train_epochs=50 \
  --gradient_accumulation_steps=1  \
  --learning_rate=5e-5  \
  --logging_steps=200  \
  --save_steps=25 \
  --adam_epsilon=1e-8  \
  --warmup_steps=0  \
  --dropout_rate=0.1  \
  --weight_decay=0.0  \
  --seed=42  \
  --max_grad_norm=1.0  \
  --do_train \



