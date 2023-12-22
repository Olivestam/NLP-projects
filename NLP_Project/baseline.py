import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F
import tagger
import parser_default
import parser_dynamic
import parser_hybrid


class Dataset():

    ROOT = ('<root>', '<root>', 0)  # Pseudo-root

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'rt', encoding='utf-8') as lines:
            tmp = [Dataset.ROOT]
            for line in lines:
                if not line.startswith('#'):  # Skip lines with comments
                    line = line.rstrip()
                    if line:
                        columns = line.split('\t')
                        if columns[0].isdigit():  # Skip range tokens
                            tmp.append((columns[1], columns[3], int(columns[6])))
                    else:
                        yield tmp
                        tmp = [Dataset.ROOT]

def make_vocabs(gold_data):
    vocab_words = {}
    vocab_tags = {}
    vocab_words['<pad>'] = len(vocab_words)
    vocab_tags['<pad>'] = len(vocab_tags)
    vocab_words['<unk>'] = len(vocab_words)
    for data in gold_data:
      for word in data:
        if word[0] not in vocab_words:
          vocab_words[word[0]] = len(vocab_words)
        if word[1] not in vocab_tags:
          vocab_tags[word[1]] = len(vocab_tags)

    #Make reverse vocab_tags
    vocab_tags_reverse = {}
    for key, value in vocab_tags.items():
      vocab_tags_reverse[value] = key

    return vocab_words, vocab_tags, vocab_tags_reverse

def main():
    # Load training data and development data
    train_data = Dataset('en_ewt-ud-train-projectivize.conllu')
    print(len(list(train_data)))
    dev_data = Dataset('en_ewt-ud-dev-projectivize.conllu')

    vocab_words, vocab_tags, vocab_tags_reverse = make_vocabs(list(train_data))
    #acc = tagger.run_tagger(vocab_words, vocab_tags, vocab_tags_reverse, train_data, dev_data)
    #print('Accuracy for tagger:', acc)

    #train_data = Dataset('en_ewt-ud-train-retagged.conllu')

    acc_parser = parser_default.run_parser(vocab_words, vocab_tags, train_data, dev_data)
    #print('Accuracy for parser:', acc_parser)

main()