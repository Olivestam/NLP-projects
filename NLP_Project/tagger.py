import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F

class Dataset():

    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'rt', encoding='utf-8') as lines:
            tmp = []
            for line in lines:
                if not line.startswith('#'):  # Skip lines with comments
                    line = line.rstrip()
                    if line:
                        columns = line.split('\t')
                        if columns[0].isdigit():  # Skip range tokens
                            tmp.append(columns)
                    else:
                        yield tmp
                        tmp = []

class FixedWindowModel(nn.Module):
    def __init__(self, embedding_specs, hidden_dim = 100, output_dim=100):
        super().__init__()
        self.embedding_layers = nn.ModuleList([])
        self.linear_input = 0
        for specs in embedding_specs:
            embedding = nn.Embedding(specs[1], specs[2])
            embedding.weight.data.normal_(0.0,0.01)
            self.embedding_layers.append(embedding)
            self.linear_input += specs[0]*specs[2]
        self.instances = [value[0] for value in embedding_specs]
        self.hidden_linear = nn.Linear(self.linear_input, hidden_dim, bias = True)
        self.relu = torch.relu
        self.linear = nn.Linear(hidden_dim, output_dim, bias = True)

    def forward(self, features):
        index = 0
        embedd = []
        for i in range(len(self.embedding_layers)):
          embedd.append(self.embedding_layers[i](features[:,index:index+self.instances[i]].long()))
          index += self.instances[i]
          if self.instances[i] > 1 :
            embedd[i] = embedd[i].view(len(features),1,-1)
        concat = torch.cat(tuple(embedd), dim=2)
        print(concat.shape)
        hidden = self.hidden_linear(concat)
        print(hidden.shape)
        relu = self.relu(hidden)
        print(relu.shape)
        output = self.linear(relu)
        print(output.shape)
        return output

class FixedWindowTagger(object):

    def __init__(self, vocab_words, vocab_tags, word_dim=50, tag_dim=10, hidden_dim=100):
        self.embedding_specs = [(3, len(vocab_words), word_dim), (1, len(vocab_tags), tag_dim)]
        self.model = FixedWindowModel(self.embedding_specs, hidden_dim, len(vocab_tags))
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags

    def featurize(self, words, i, pred_tags):
        output = torch.zeros(4)
        output[0] = words[i]
        if i == 0:
          output[1] = self.vocab_words['<pad>']
          output[3] = self.vocab_tags['<pad>']
          if len(words)<2:
            output[2] = self.vocab_words['<pad>']
          else:
            output[2] = words[i+1]
        elif len(words) == (i+1):
          output[1] = words[i-1]
          output[3] = pred_tags[i-1]
          output[2] = self.vocab_words['<pad>']
        else:
          output[1] = words[i-1]
          output[3] = pred_tags[i-1]
          output[2] = words[i+1]
        return output

    def predict(self, words, vocab_tags_reverse):
        pred_tags = torch.zeros(len(words))
        words_id = torch.zeros(len(words))
        for i in range(len(words)):
          if words[i] in self.vocab_words:
            words_id[i] = self.vocab_words[words[i]]
          else:
            words_id[i] = self.vocab_words['<unk>']
        for i in range(len(words_id)):
          features = self.featurize(words_id, i, pred_tags)
          features = features.unsqueeze(0)
          tag = self.model.forward(features)
          index_tag = tag.argmax()
          pred_tags[i] = index_tag.item()
        pred_tags_list = []
        for i in range(len(pred_tags)):
          pred_tags_list.append(vocab_tags_reverse[int(pred_tags[i].item())])

        return pred_tags_list

def accuracy(tagger, gold_data, vocab_tags_reverse):
    # TODO: Replace the next line with your own code
    correct = 0
    total = 0
    for gold in gold_data:
      words = [j[0] for j in gold]
      predict = tagger.predict(words, vocab_tags_reverse)
      for j in range(len(predict)):
        if predict[j] == gold[j][1]:
          correct +=1
        total += 1
    return (correct/total)

def training_examples(vocab_words, vocab_tags, gold_data, tagger, batch_size=100):
    batch = torch.zeros((batch_size,4))
    tags = torch.zeros(batch_size)
    index = 0
    for data in gold_data:
      words_id = torch.zeros(len(data))
      tags_id = torch.zeros(len(data))
      for i, word in enumerate(data):
        words_id[i] = vocab_words[word[0]]
        tags_id[i] = vocab_tags[word[1]]
      for i in range(len(words_id)):
        row = tagger.featurize(words_id, i, tags_id)
        tags[index] = tags_id[i]
        batch[index] = row
        if index == (batch_size - 1):
          yield batch, tags
          index = -1
          batch = torch.zeros((batch_size,4))
          tags = torch.zeros(batch_size)
        index += 1
    if index < batch_size:
      batch = batch[0:index]
      tags = tags[0:index]
      yield batch, tags

def train_fixed_window(train_data, vocab_words, vocab_tags, n_epochs=2, batch_size=100, lr=1e-2):
    print('Runing training for tagger...')

    # Initialize the model and tagger
    tagger = FixedWindowTagger(vocab_words, vocab_tags)
       
    # Initialize the optimizer. Here we use Adam rather than plain SGD
    optimizer = optim.Adam(tagger.model.parameters(), lr=lr)

    for epoch in range(n_epochs): 
        print('Epoch:', (epoch+1), '/', n_epochs)
        tagger.model.train()
        for batch, tags in training_examples(vocab_words, vocab_tags, train_data, tagger, batch_size):
            optimizer.zero_grad()
            output = tagger.model.forward(batch)
            output = output.squeeze(1)
            print(output.shape)
            print(tags.shape)
            tags = tags.long()
            loss = F.cross_entropy(output, tags)
            loss.backward()
            optimizer.step()
    return tagger

def run_tagger(vocab_words, vocab_tags, vocab_tags_reverse, train_data, dev_data):
   # Train the tagger
   trained_tagger = train_fixed_window(train_data, vocab_words, vocab_tags)

   # Calculate accuracy
   acc = accuracy(trained_tagger, dev_data, vocab_tags_reverse)

   with open('en_ewt-ud-train-retagged.conllu', 'wt', encoding='utf-8') as target:
    for sentence in Dataset('en_ewt-ud-train-projectivize.conllu'):
        words = [columns[1] for columns in sentence]
        for i, t in enumerate(trained_tagger.predict(words, vocab_tags_reverse)):
            sentence[i][3] = t
        for columns in sentence:
            print('\t'.join(c for c in columns), file=target)
        print(file=target)

   return acc

