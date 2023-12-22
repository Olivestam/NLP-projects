import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class FixedWindowModel(nn.Module):
    def __init__(self, hidden_dim = 100, output_dim=100):
        super().__init__()
        self.hidden_linear = nn.Linear(6*hidden_dim, hidden_dim, bias = True)
        self.linear = nn.Linear(hidden_dim, output_dim, bias = True)

    def forward(self, features):
        input_hidden = torch.cat([features[:, i, :] for i in range(features.size(1))], dim=1)
        hidden = self.hidden_linear(input_hidden)
        relu = F.relu(hidden)
        output = self.linear(relu)
        return output
    
class FixedWindowModelLSTM(nn.Module):
    def __init__(self, word_dim, tag_dim, vocab_words, vocab_tags, hidden_dim = 100):
        super().__init__()
        self.embedding_word = nn.Embedding(len(vocab_words), word_dim)
        self.embedding_tag = nn.Embedding(len(vocab_tags), tag_dim)
        self.embedding_word.weight.data.normal_(0.0,0.01)
        self.embedding_tag.weight.data.normal_(0.0,0.01)
        self.rnn = nn.LSTM((word_dim+tag_dim), hidden_dim, bidirectional = True, batch_first = True, bias = True)

    def forward(self, sentence, tags):
        embedd = []
        rnn_out = []
        for i in range(len(sentence)):
           word_embedd = self.embedding_word(sentence[i])
           tag_embedd = self.embedding_tag(tags[i])
           concat = torch.cat((word_embedd, tag_embedd), dim=0)
           embedd.append(concat)
        """ for input in embedd:
          input = input.unsqueeze(0)
          rnn, _ =  self.rnn(input)
          rnn = rnn.squeeze(0)
          rnn_out.append(rnn) """
        embedd = torch.vstack(embedd).unsqueeze(0)
        rnn_out, _ = self.rnn(embedd)
        #print(rnn_out.shape)
        return rnn_out.squeeze(0)

class ArcStandardParser(object):

    MOVES = tuple(range(3))

    SH, LA, RA = MOVES  # Parser moves are specified as integers.

    @staticmethod
    def initial_config(num_words):
        i = 0
        stack = []
        heads = [0 for _ in range(num_words)]
        return (i, stack, heads)
        

    @staticmethod
    def valid_moves(config):
        valid = [0,1,2]
        if len(config[1]) <= 1:
          del valid[2]
        if len(config[1]) == 0:
          del valid[1]
        if config[0] >= len(config[2]):
          if len(valid) > 1:
            del valid[1]
          del valid[0]
        return valid

    @staticmethod
    def next_config(config, move):
        i = config[0]
        stack = config[1]
        heads = config[2]
        if move == 0:
          stack.append(i)
          i += 1
        elif move == 1:
          heads[stack[-1]] = i
          del stack[-1]
        elif move == 2:
          heads[stack[-1]] = stack[-2]
          del stack[-1]
        return (i, stack, heads)
    

    @staticmethod
    def is_final_config(config):
        if len(ArcStandardParser.valid_moves(config)) == 0:
          return True
        return False


class FixedWindowParser(ArcStandardParser):

    def __init__(self, vocab_words, vocab_tags, word_dim=100, tag_dim=25, hidden_dim=100):
        self.lstm_model = FixedWindowModelLSTM(word_dim, tag_dim, vocab_words, vocab_tags, hidden_dim)
        self.model = FixedWindowModel(hidden_dim, 3)
        self.vocab_words = vocab_words
        self.vocab_tags = vocab_tags

    def featurize(self, rnn_embeddings, config):
        output = torch.zeros((3, len(rnn_embeddings[0])))
        # Add next word and tag in buffer
        if config[0] < len(rnn_embeddings):
            output[2] = rnn_embeddings[config[0]]
        else:
            output[2] = self.vocab_words['<pad>']
        # Add top most and second top words and their tags from stack
        if len(config[1]) > 1:
            output[0] = rnn_embeddings[config[1][-1]]
            output[1] = rnn_embeddings[config[1][-2]]
        elif len(config[1]) > 0:
            output[0] = rnn_embeddings[config[1][-1]]
            output[1] = self.vocab_words['<pad>']
        else:
            output[0] = self.vocab_words['<pad>']
            output[1] = self.vocab_words['<pad>']
        return output

    def predict(self, words, tags):
        words_id = torch.zeros(len(words))
        tags_id = torch.zeros(len(tags))
        for i in range(len(words)):
          tags_id[i] = self.vocab_tags[tags[i]]
          if words[i] in self.vocab_words:
            words_id[i] = self.vocab_words[words[i]]
          else:
            words_id[i] = self.vocab_words['<unk>']
        config = self.initial_config(len(words))
        words_id = words_id.long()
        tags_id = tags_id.long()
        rnn_embeddings = self.lstm_model.forward(words_id, tags_id)
        while not self.is_final_config(config):
          features = self.featurize(rnn_embeddings, config)
          features = features.unsqueeze(0)
          moves = self.model.forward(features)
          moves = moves.squeeze(0)
          moves = moves.squeeze(0)
          best_move = None
          highest_score = - math.inf
          valid_moves = self.valid_moves(config)
          for i in range(len(moves)):
            if moves[i] > highest_score:
              if i in valid_moves:
                highest_score = moves[i]
                best_move = i
          config = self.next_config(config, best_move)
        return config[2]
  
def uas(parser, gold_data):
    correct = 0
    total = 0
    for data in gold_data:
      predict = parser.predict([t[0] for t in data], [t[1] for t in data])
      gold = [t[2] for t in data]
      correct += sum(a == b for a, b in zip(predict[1:len(predict)], gold[1:len(gold)]))
      total += len(predict)-1
    return correct/total
    
def oracle_moves(gold_heads):
    config = ArcStandardParser.initial_config(len(gold_heads))

    for i in range(2*len(gold_heads)-1):
      valid_moves = ArcStandardParser.valid_moves(config)
      #print(valid_moves, config[1], config[2],gold_heads)
      if (len(valid_moves) == 1) and (valid_moves[0] == 0):
        yield config, valid_moves[0]
        config =  ArcStandardParser.next_config(config, valid_moves[0])

      if len(valid_moves) == 2:
        #print('LA',config[1], valid_moves)
        # choose LA
        counterGoldSecond = [1 for i in gold_heads if i == config[1][-1]]
        counterHeadsSecond = [1 for i in config[2] if i == config[1][-1]]

        if (len(counterGoldSecond) == len(counterHeadsSecond)) and (gold_heads[config[1][-1]] == config[0]):
          #print('LA')
          yield config, 1
          config =  ArcStandardParser.next_config(config, 1)

        else:
          #print('SH')
          yield config, 0
          config =  ArcStandardParser.next_config(config, 0)
         
      if 2 in valid_moves:
        #print('RA',config[1], valid_moves)
        # choose LA
        counterGoldSecond = [1 for i in gold_heads if i == config[1][-1]]
        counterHeadsSecond = [1 for i in config[2] if i == config[1][-1]]

          # choose RA
        counterGoldTop = [1 for i in gold_heads if i ==  config[1][-1]]
        counterHeadsTop = [1 for i in config[2] if i ==  config[1][-1]]
          
        if (len(counterGoldSecond) == len(counterHeadsSecond)) and (gold_heads[config[1][-1]] == config[0]):
          #print('LA')
          yield config, 1
          config =  ArcStandardParser.next_config(config, 1)
            
        elif (len(counterGoldTop) == len(counterHeadsTop)) and (gold_heads[config[1][-1]] == config[1][-2]):
          #print('RA')
          yield config, 2
          config =  ArcStandardParser.next_config(config, 2)
            
        else:
          #print('SH')
          yield config, 0
          config =  ArcStandardParser.next_config(config, 0)

def training_examples(vocab_words, vocab_tags, gold_data, parser, batch_size=100):
    for data in gold_data:
      index = 0
      batch = torch.zeros(((len(data)*2-1), 3, 200))
      moves = torch.zeros((len(data)*2-1))
      gold_heads = [t[2] for t in data]
      words_id = torch.zeros(len(data))
      tags_id = torch.zeros(len(data))
      for i, word in enumerate(data):
        if word[0] in vocab_words:
          words_id[i] = vocab_words[word[0]]
        else:
          words_id[i] = vocab_words['<unk>']
        tags_id[i] = vocab_tags[word[1]]
      words_id = words_id.long()
      tags_id = tags_id.long()
      rnn_embeddings = parser.lstm_model.forward(words_id, tags_id)
      config = parser.initial_config(len(words_id))
      for config, move in oracle_moves(gold_heads):
        row = parser.featurize(rnn_embeddings, config)
        moves[index] = move
        batch[index] = row
        index += 1
      yield batch, moves

def train_fixed_window(train_data,vocab_words, vocab_tags, n_epochs=1, batch_size=100, lr=1e-2):
    print('Started training parser...')
     # Initialize the model and tagger
    parser = FixedWindowParser(vocab_words, vocab_tags)

    # Initialize the optimizer. Here we use Adam rather than plain SGD
    optimizer = optim.Adam(list(parser.model.parameters()) + list(parser.lstm_model.parameters()), lr=lr)

    for epoch in range(n_epochs):
        print('Epoch', (epoch+1), '/', n_epochs)
        losses = []
        parser.model.train()
        with tqdm(total=12544) as pbar:
          for batch, moves in training_examples(vocab_words, vocab_tags, train_data, parser, batch_size):
              optimizer.zero_grad()
              output = parser.model.forward(batch)
              output = output.squeeze(1)
              moves = moves.long()
              loss = F.cross_entropy(output, moves)
              loss.backward()
              optimizer.step()
              losses.append(loss.item())
              pbar.set_postfix(loss = (sum(losses)/len(losses)))
              pbar.update(1)
    return parser

def run_parser(vocab_words, vocab_tags, train_data, dev_data):
   # Train the parser
   trained_parser = train_fixed_window(train_data, vocab_words, vocab_tags)

   # Calculate accuracy
   acc = uas(trained_parser, dev_data)

   return acc