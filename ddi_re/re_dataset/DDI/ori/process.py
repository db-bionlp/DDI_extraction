from transformers import BasicTokenizer
tokenizer = BasicTokenizer(do_lower_case=True, never_split=None, tokenize_chinese_chars=True, strip_accents=None)
def process_result(input_filename, output_filename):
    print("Reading {} file".format(input_filename))
    print("Reading {} file".format(output_filename))
    input = open(input_filename, 'r', encoding='utf-8')
    output= open(output_filename, "w", encoding="utf-8")

    samples = input.read().strip().split('\n\n')
    for sample in samples:
        index_sent_pairs = sample.split("\n")
        id_sent = index_sent_pairs[0]
        sent = id_sent.split("\t")
        if len(sent)==1 :
            continue
        else:
            sent=sent[1]
        pairs = index_sent_pairs[1:]
        output.write(sent+"\n")
        for pair in pairs:
            # print(pair.split("\t"))
            drug1, type1, span1, drug2, type2, span2 ,re = pair.split("\t")
            output.write(drug1+"\t"+span1+"\t"+ drug2+"\t"+span2+"\t"+"\n")
        output.write("\n")


# process_result("load_train_data.txt", "../train.tsv")
# process_result("load_dev_data.txt", "../dev.tsv")
process_result("load_test_data.txt", "../test.tsv")