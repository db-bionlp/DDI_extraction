import copy, os, sys
import json
import logging
import re
from torch.utils.data import TensorDataset
logger = logging.getLogger(__name__)
sys.path.append('../')
from ddi_re.re_task.utils import *
from transformers import BertTokenizer
from scipy.stats import chi2
from scipy.stats import t as t_func

class InputFeatures(object):

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, center_list, div_list,
                ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.center_list = center_list
        self.div_list = div_list

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputtestFeatures(object):

    def __init__(self, input_ids, attention_mask, token_type_ids, center_list, div_list,
                ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.center_list = center_list
        self.div_list = div_list

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def read_csv(args,data_filename):
    print("Reading {} file".format(data_filename))
    fp = open(data_filename, 'r', encoding='utf-8')
    samples = fp.read().strip().split('\n\n')
    sent_contents = []
    sent_lables = []
    drug_pairs=[]
    labels = get_label(args)
    for sample in samples:
        sent_re = sample.split('\n')
        sent = sent_re[0]
        for pair in sent_re[1:]:

            drug1, span1, drug2, span2, relation = pair.split("\t")
            try:
                span1_start = int(span1.split("-")[0])
                span1_end = int(span1.split("-")[1])
                span2_start = int(span2.split("-")[0])
                span2_end = int(span2.split("-")[1])
                tag1 = "&" * (span1_end - span1_start + 1)
                tag2 = "#" * (span2_end - span2_start + 1)
                sent_copy = sent[:span1_start] + tag1 + sent[span1_end + 1:span2_start] + tag2 + sent[span2_end + 1:]
                sent_copy = re.sub(r'&*&', "@DRUG$", sent_copy)
                sent_copy = re.sub(r'#*#', "@DRUG$", sent_copy)
                sent_contents.append(sent_copy)
                label = labels.index(relation)
                sent_lables.append(label)
                drug_pairs.append((drug1, drug2))

            except:
                continue

    return sent_contents, sent_lables

def read_test_csv(args, data_filename):
    print("Reading {} file".format(data_filename))
    fp = open(data_filename, 'r', encoding='utf-8')
    samples = fp.read().strip().split('\n\n')
    sent_contents = []
    drug_pairs=[]
    for sample in samples:
        sent_re = sample.split('\n')
        sent = sent_re[0]
        for pair in sent_re[1:]:
            # print(pair)
            try:
                drug1, span1, drug2, span2, = pair.split("\t")[:4]
                span1_start = int(span1.split("-")[0])
                span1_end = int(span1.split("-")[1])
                span2_start = int(span2.split("-")[0])
                span2_end = int(span2.split("-")[1])
                tag1 = "&" * (span1_end - span1_start + 1)
                tag2 = "#" * (span2_end - span2_start + 1)
                sent_copy = sent[:span1_start] + tag1 + sent[span1_end + 1:span2_start] + tag2 + sent[span2_end + 1:]
                sent_copy = re.sub(r'&*&', "@DRUG$", sent_copy)
                sent_copy = re.sub(r'#*#', "@DRUG$", sent_copy)
                sent_contents.append(sent_copy)
                drug_pairs.append((drug1, drug2))
            except:
                continue
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    test_drug_pair_file = os.path.join(args.model_dir, "test_drug_pairs.tsv")
    with open(test_drug_pair_file, "w", encoding="utf-8") as f:
        for drug_pair in drug_pairs:
            for drug in drug_pair:
                f.write(str(drug)+"\t")
            f.write("\n")

    return sent_contents

def load_tokenize(args):
    # ADDITIONAL_SPECIAL_TOKENS = ["DRUG1", "DRUG2", "DRUGOTHER"]
    ADDITIONAL_SPECIAL_TOKENS = ["@DRUG$"]
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer

def makeFeatures(args,sent_list,sent_labels,mode):
    index = 0
    maxlength = 300
    print('Making {} Features'.format(mode))
    features = []
    tokenizer = load_tokenize(args)
    for sent, label in zip(sent_list, sent_labels):
        index += 1
        count=0
        label = int(label)
        tokens_a = tokenizer.tokenize(sent)
        special_tokens_count = 2
        if len(tokens_a) > maxlength - special_tokens_count:
            tokens_a = tokens_a[:(maxlength - special_tokens_count)]
        tokens = ['CLS'] + tokens_a + ['SEP']
        enity_index = []

        word_count = 0
        for word in tokens:
            if word =="@DRUG$":
                enity_index.append(word_count)
            word_count += 1
        # enity_index.append(tokens.index("DRUG1"))
        # enity_index.append(tokens.index("DRUG2"))

        # Generate the Center vector
        t = 0.1
        if len(enity_index) == 2:
            dis = (enity_index[1] - enity_index[0] + 1) / len(tokens)
            center_list = [0] * (len(tokens))
            for i in range(len(tokens)):
                center_list[i] = random.uniform(0.3-t*dis, 0.5+t*dis)
            for i in range(enity_index[0], enity_index[1] + 1):
                center_list[i] = random.uniform(0.9-t*dis, 1.1+t*dis)
        else:
            center_list = [1] * (len(tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        token_type = [0] * len(tokens)
        attention_mask = [1] * len(tokens)

        # Zero-pad up to the sequence length.
        padding_length = maxlength - len(tokens)

        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type = token_type + ([0] * padding_length)
        center_list = center_list +([0] * padding_length)

        # Vector for filling
        pad_list = [1]*len(center_list)

        if args.use_Under_sampling_and_over_sampling:
            # over_sampling operation
            if label != 0 and mode =='train':
                # Generate the Diversified vector
                chi2_list = []
                n = 2
                for i in range(7):
                    j = i + 1
                    chi2_list.append(10 * (chi2.cdf(j, n) - chi2.cdf(i, n)))

                t_list = []
                for i in range(7):
                    j = i + 1
                    t_list.append((t_func.cdf(j, 10) - t_func.cdf(i, 10)))

                if len(enity_index) == 2:
                    chi2_list1 = [1] * len(tokens)
                    t_list1 = [1] * len(tokens)

                    count = 0
                    for i in range(int(enity_index[0] - 0.15 * len(tokens)), enity_index[0]):
                        if i >= 0:
                            if count < len(chi2_list):
                                count += 1
                                chi2_list1[i] = chi2_list[count - 1]
                            else:
                                chi2_list1[i] = chi2_list[count - 1]
                    count = 0
                    for i in range(enity_index[0], int(enity_index[0] + 0.25 * len(tokens))):
                        if i < len(tokens):
                            if count < len(chi2_list):
                                chi2_list1[i] = chi2_list[len(chi2_list) - count - 1]
                                count += 1
                            else:
                                chi2_list1[i] = chi2_list[len(chi2_list) - count - 1]
                    count = 0
                    for i in range(int(enity_index[1] - 0.25 * len(tokens)), enity_index[1]):
                        if i > 0:
                            if count < len(chi2_list):
                                count += 1
                                chi2_list1[i] = chi2_list[count - 1]
                            else:
                                chi2_list1[i] = chi2_list[count - 1]
                    count = 0
                    for i in range(enity_index[1], int(enity_index[1] + 0.1 * len(tokens))):
                        if i < len(tokens):
                            if count < len(chi2_list):
                                count += 1
                                chi2_list1[i] = chi2_list[count - 1]
                            else:
                                chi2_list1[i] = chi2_list[count - 1]

                    count = 0
                    for i in range(int(enity_index[0] - 0.15 * len(tokens)), enity_index[0]):
                        if i >= 0:
                            if count < len(t_list):
                                count += 1
                                t_list1[i] = t_list[count - 1]
                            else:
                                t_list1[i] = t_list[count - 1]
                    count = 0
                    for i in range(enity_index[0], int(enity_index[0] + 0.25 * len(tokens))):
                        if i < len(tokens):
                            if count < len(t_list):
                                t_list1[i] = t_list[len(t_list) - count - 1]
                                count += 1
                            else:
                                t_list1[i] = t_list[len(t_list) - count - 1]
                    count = 0
                    for i in range(int(enity_index[1] - 0.25 * len(tokens)), enity_index[1]):
                        if i > 0:
                            if count < len(t_list):
                                count += 1
                                t_list1[i] = t_list[count - 1]
                            else:
                                t_list1[i] = t_list[count - 1]
                    count = 0
                    for i in range(enity_index[1], int(enity_index[1] + 0.1 * len(tokens))):
                        if i < len(tokens):
                            if count < len(t_list):
                                count += 1
                                t_list1[i] = t_list[count - 1]
                            else:
                                t_list1[i] = t_list[count - 1]

                else:

                    chi2_list1 = [1] * len(tokens)
                    t_list1 = [1] * len(tokens)
                chi2_list1 = chi2_list1 + ([0] * padding_length)
                t_list1 = t_list1 + ([0] * padding_length)


                features.append(
                    InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type,
                                  label_id=label,
                                  center_list=center_list,
                                  div_list=chi2_list1,

                                  ))

                # show examples
                if index < 4 and mode == 'train' and label != 0:
                    logging.info("*** Example ***")
                    logging.info("tokens: %s", " ".join([str(x) for x in tokens]))
                    logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                    logging.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
                    logging.info("token_type_ids: %s", " ".join([str(x) for x in token_type]))
                    logging.info("label: %d", label)
                    logging.info("center_list: %s", " ".join([str(x) for x in center_list]))
                    logging.info("diversified_list_for_Chi_distribution: %s", " ".join([str(x) for x in chi2_list1]))
                    # logging.info("diversified_list_for_T_distribution: %s", " ".join([str(x) for x in t_list1]))


            # other
            else:
                features.append(
                    InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type,
                                  label_id=label,
                                  center_list=center_list,
                                  div_list=pad_list,

                                  ))
                if index < 4 :
                    logging.info("*** Example ***")
                    logging.info("tokens: %s", " ".join([str(x) for x in tokens]))
                    logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                    logging.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
                    logging.info("token_type_ids: %s", " ".join([str(x) for x in token_type]))
                    logging.info("label: %d", label)
                    logging.info("center_list: %s", " ".join([str(x) for x in center_list]))
                    logging.info("diversified_list: %s", " ".join([str(x) for x in pad_list]))


        # not use U_O_sampling
        else:
            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type,
                              label_id=label,
                              center_list=center_list,
                              div_list=pad_list,

                              ))
            if index < 4:
                logging.info("*** Example ***")
                logging.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logging.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
                logging.info("token_type_ids: %s", " ".join([str(x) for x in token_type]))
                logging.info("label: %d", label)
                logging.info("center_list: %s", " ".join([str(x) for x in center_list]))

    return features

def maketestFeatures(args,sent_list,mode):
    index = 0
    maxlength = 300
    print('Making {} Features'.format(mode))
    features = []
    tokenizer = load_tokenize(args)
    for sent in sent_list:
        index += 1
        count=0
        tokens_a = tokenizer.tokenize(sent)
        special_tokens_count = 2
        if len(tokens_a) > maxlength - special_tokens_count:
            tokens_a = tokens_a[:(maxlength - special_tokens_count)]
        tokens = ['CLS'] + tokens_a + ['SEP']
        enity_index = []

        word_count = 0
        for word in tokens:
            if word =="@DRUG$":
                enity_index.append(word_count)
            word_count += 1
        # enity_index.append(tokens.index("DRUG1"))
        # enity_index.append(tokens.index("DRUG2"))

        # Generate the Center vector
        t = 0.1
        if len(enity_index) == 2:
            dis = (enity_index[1] - enity_index[0] + 1) / len(tokens)
            center_list = [0] * (len(tokens))
            for i in range(len(tokens)):
                center_list[i] = random.uniform(0.3-t*dis, 0.5+t*dis)
            for i in range(enity_index[0], enity_index[1] + 1):
                center_list[i] = random.uniform(0.9-t*dis, 1.1+t*dis)
        else:
            center_list = [1] * (len(tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        token_type = [0] * len(tokens)
        attention_mask = [1] * len(tokens)

        # Zero-pad up to the sequence length.
        padding_length = maxlength - len(tokens)

        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type = token_type + ([0] * padding_length)
        center_list = center_list +([0] * padding_length)

        # Vector for filling
        pad_list = [1]*len(center_list)



        features.append(
                InputtestFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type,
                              center_list=center_list,
                              div_list=pad_list,

                              ))
        if index < 4:
                logging.info("*** Example ***")
                logging.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logging.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logging.info("attention_mask: %s", " ".join([str(x) for x in attention_mask]))
                logging.info("token_type_ids: %s", " ".join([str(x) for x in token_type]))
                logging.info("center_list: %s", " ".join([str(x) for x in center_list]))

    return features


def load_and_cache_examples(args, mode):
    # Load data features from cache or re_dataset file
    if args.use_Under_sampling_and_over_sampling:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}".format(
                mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(), ),
        )
    else:
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}".format(
                mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(), ),
        )

    if os.path.exists(cached_features_file) :
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logger.info("Creating features from re_dataset file at %s", args.data_dir)

        if mode == "train":
            sent_contents, sent_lables = read_csv(args, args.train_filename)
            features = makeFeatures(args, sent_contents, sent_lables,
                                  mode)
        elif mode == "dev":
            sent_contents, sent_lables = read_csv(args, args.dev_filename)
            features = makeFeatures(args, sent_contents, sent_lables,  mode)
        elif mode == "test":
            sent_contents = read_test_csv(args, args.test_filename)
            features = maketestFeatures(args, sent_contents, mode)
        else:
            raise Exception("For mode, Only train,dev,test is available")

        if mode != 'test':
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Convert to Tensors and build re_dataset
    if mode !="test":
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        all_center_list = torch.tensor([f.center_list for f in features], dtype=torch.float)
        all_div_list = torch.tensor([f.div_list for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                all_center_list,
                                all_div_list,
                                all_label_ids,
                                )
    else:
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_center_list = torch.tensor([f.center_list for f in features], dtype=torch.float)
        all_div_list = torch.tensor([f.div_list for f in features], dtype=torch.float)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                                all_center_list,
                                all_div_list,
                                )

    return dataset




