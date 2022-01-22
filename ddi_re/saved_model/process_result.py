def process_result(pairs_filename, re_filename, result):
    print("Reading {} file".format(pairs_filename))
    print("Reading {} file".format(re_filename))
    pairs = open(pairs_filename, 'r', encoding='utf-8')
    re = open(re_filename, 'r', encoding='utf-8')

    pair_samples = pairs.read().strip().split('\n')
    re_samples = re.read().strip().split('\n')

    if len(pair_samples) == len(re_samples):
        with open(result, "w", encoding="utf-8") as f:
            for pair, re in zip(pair_samples, re_samples):
                f.write(pair+re+"\n")
process_result("DDI/test_drug_pairs.tsv", "DDI/test_output.tsv", "DDI/result.tsv")