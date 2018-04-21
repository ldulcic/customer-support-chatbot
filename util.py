from constants import CUDA


def cuda(input):
    if CUDA:
        return input.cuda()
    else:
        return input


# convert twitter customer support dataset from 1-line Q 1-line A to tsv (Q <tab> A)
def txt2tsv(txt_path, tsv_path):
    with open(txt_path, encoding='utf-8') as txt:
        lines = txt.readlines()
    qa_pair_lines = ["%s\t%s\n" % (lines[i].rstrip(), lines[i + 1].rstrip()) for i in range(0, len(lines), 2)]
    with open(tsv_path, 'w', encoding='utf-8') as tsv:
        tsv.writelines(qa_pair_lines)


if __name__ == '__main__':
    txt2tsv('data/twitter_customer_support/twitter_customer_support.txt',
            'data/twitter_customer_support/twitter_customer_support.tsv')
