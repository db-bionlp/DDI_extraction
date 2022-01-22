from transformers import BasicTokenizer
tokenizer = BasicTokenizer(do_lower_case=False, never_split=None, tokenize_chinese_chars=True, strip_accents=None)
def process_result(input_filename, output_filename):
    print("Reading {} file".format(input_filename))
    print("Reading {} file".format(output_filename))
    input = open(input_filename, 'r', encoding='utf-8')
    output= open(output_filename, "w", encoding="utf-8")
    # out = open(out_filename, "w", encoding="utf-8")

    samples = input.read().strip().split('\n\n')
    for sample in samples:
        index_sent_pairs = sample.split("\n")
        id_sent = index_sent_pairs[0]
        sent = id_sent.split("\t")
        if len(sent) == 1:
            continue
        else:
            sent = sent[1]
        sent = tokenizer.tokenize(text=sent)
        pairs = index_sent_pairs[1:]
        # output.write(sent + "\n")
        col =[]
        for pair in pairs:
            # print(pair.split("\t"))
            drug1, type1, span1, drug2, type2, span2, re = pair.split("\t")
            col.append(drug1)
            col.append(drug2)

            drug1 = drug1.split(" ")
            for a in drug1:
                col.append(a)

            drug2 = drug2.split(" ")
            for a in drug2:
                col.append(a)

        word_last=""
        for word in sent:
            if word in col:
                if word_last=="B" or word_last=="I":
                    output.write(word + "\t" + "I" + "\n")
                    word_last="I"
                else:
                    output.write(word + "\t" + "B" + "\n")
                    word_last = "B"
            else:
                output.write(word + "\t" + "O" + "\n")
                word_last = "O"


        output.write("\n")

def process_test_result(input_filename, output_filename):
    print("Reading {} file".format(input_filename))
    print("Reading {} file".format(output_filename))
    input = open(input_filename, 'r', encoding='utf-8')
    output= open(output_filename, "w", encoding="utf-8")
    # out = open(out_filename, "w", encoding="utf-8")

    samples = input.read().strip().split('\n\n')
    for sample in samples:
        index_sent_pairs = sample.split("\n")
        id_sent = index_sent_pairs[0]
        sent = id_sent.split("\t")
        if len(sent) == 1:
            continue
        else:
            sent = sent[1]
        sent = tokenizer.tokenize(text=sent)
        pairs = index_sent_pairs[1:]

        for word in sent:
            output.write(word +"\n")
        output.write("\n")




# process_test_result("load_test_data.txt", "../test.tsv")
# process_result("load_dev_data.txt", "../devel.tsv")
process_result("load_train_data.txt", "../train.tsv")