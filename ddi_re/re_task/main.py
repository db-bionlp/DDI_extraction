import os
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
import argparse
import sys
sys.path.append('../')
sys.path.append('./')
from ddi_re.re_task.utils import MODEL_CLASSES, MODEL_PATH_MAP, init_logger
from ddi_re.re_task.trainer import Trainer
from ddi_re.re_task.load_data import load_and_cache_examples

def main(args):
    init_logger()

    train_dataset = load_and_cache_examples(args, mode='train')

    dev_dataset = load_and_cache_examples(args, mode='dev')

    test_dataset = load_and_cache_examples(args, mode="test")


    trainer = Trainer(args,train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset)

    if args.do_train:
        trainer.train()

    # if args.do_eval:
    #     trainer.load_model()
    #     trainer.evaluate('dev')
    #
    # if args.do_test:
    #     trainer.load_model()
    #     trainer.predict('test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_filename", default='../re_dataset/DDI/train.tsv', type=str, help="train file name")
    parser.add_argument("--dev_filename", default='../re_dataset/DDI/dev.tsv', type=str, help=" test file name")
    parser.add_argument("--test_filename", default='../re_dataset/DDI/test.tsv', type=str,  help=" dev file name")
    parser.add_argument("--data_dir",  default='../re_dataset/DDI', type=str, help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--label_file",  default='../re_dataset/label.csv', type=str, help="Label file")
    parser.add_argument("--model_dir",  default='../saved_model/DDI', type=str, help="Path to model")
    parser.add_argument("--model_type",  default='biobert', type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model",   default='bert_center', type=str,  help="which model to use: only_bert , bert_center,character_bert_center")
    parser.add_argument("--do_train", default=True,  help="whether do train.")
    parser.add_argument("--do_eval", default=False,  help="whether do dev.")
    parser.add_argument("--do_test", default=False,  help="whether do test.")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--num_train_epochs", default=100.0, type=float,
                        help="Total number of trainingtop epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument('--logging_steps', type=int, default=250, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=250, help="Save checkpoint every X updates steps.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument('--tpu', action='store_true',
                        help="Whether to run on the TPU defined in the environment variables")
    parser.add_argument('--tpu_ip_address', type=str, default='',
                        help="TPU IP address if none are set in the environment variables")
    parser.add_argument('--tpu_name', type=str, default='',
                        help="TPU name if none are set in the environment variables")
    parser.add_argument('--xrt_tpu_config', type=str, default='',
                        help="XRT TPU config if none are set in the environment variables")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--parameter_averaging', default=False, help="Whether to use parameter averaging")
    parser.add_argument('--use_Under_sampling_and_over_sampling', default=True,
                        help="Whether to use Under_sampling_and_over_sampling ")

    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]

    main(args)