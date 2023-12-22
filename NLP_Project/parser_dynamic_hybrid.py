import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
from random import sample

class FixedWindowModel(nn.Module):
    def __init__(self, hidden_dim = 100, output_dim=100):
        super().__init__()
        """ self.embedding_layers = nn.ModuleList([])
        self.linear_input = embedding_specs[0][2] + embedding_specs[4][2]
        for specs in embedding_specs:
            embedding = nn.Embedding(specs[1], specs[2])
            embedding.weight.data.normal_(0.0,0.01)
            self.embedding_layers.append(embedding)
        self.instances = [value[0] for value in embedding_specs] """
        self.hidden_linear = nn.Linear(8*hidden_dim, hidden_dim, bias = True)
        self.relu = torch.relu
        self.linear = nn.Linear(hidden_dim, output_dim, bias = True)
        

    def forward(self, features):
        #input_hidden = features.transpose(1, 2).contiguous().view(100, -1)
        #input_hidden = torch.cat([features[:, i, :] for i in range(features.size(1))], dim=1)
        input_hidden = features.view(features.size(0), 1, 800)
        #input_hidden = features.reshape(100, -1)
        hidden = self.hidden_linear(input_hidden)
        relu = self.relu(hidden)
        output = self.linear(relu)
        return output
    
class FixedWindowModelLSTM(nn.Module):
    def __init__(self, word_dim, tag_dim, vocab_words, vocab_tags, hidden_dim = 100):
        super().__init__()
        self.embedding_word = nn.Embedding(len(vocab_words), word_dim)
        self.embedding_tag = nn.Embedding(len(vocab_tags), tag_dim)
        self.embedding_word.weight.data.normal_(0.0,0.01)
        self.embedding_tag.weight.data.normal_(0.0,0.01)
        """ self.embedding_layers = nn.ModuleList([])
        self.linear_input = embedding_specs[0][2] + embedding_specs[4][2]
        for specs in embedding_specs:
            embedding = nn.Embedding(specs[1], specs[2])
            embedding.weight.data.normal_(0.0,0.01)
            self.embedding_layers.append(embedding)
        self.instances = [value[0] for value in embedding_specs] """
        self.rnn = nn.LSTM((word_dim+tag_dim), hidden_dim, bidirectional = True, batch_first = True, bias = True)

    def forward(self, sentence, tags):
        embedd = []
        rnn_out = []
        for i in range(len(sentence)):
           word_embedd = self.embedding_word(sentence[i])
           tag_embedd = self.embedding_tag(tags[i])
           concat = torch.cat((word_embedd, tag_embedd), dim=0)
           embedd.append(concat)
        for input in embedd:
          input = input.unsqueeze(0)
          rnn, _ =  self.rnn(input)
          rnn = rnn.squeeze(0)
          rnn_out.append(rnn)
        return rnn_out

class ArcStandardParser(object):

    MOVES = tuple(range(3))

    SH, LA, RA = MOVES  # Parser moves are specified as integers.

    @staticmethod
    def zero_cost_shift(config, gold_heads):
        if config[0] >= len(config[2]):
           return False

        for i in config[1]:
           if gold_heads[i] == config[0]:
              return False
           
        return True
    
    @staticmethod
    def zero_cost_right(config, gold_heads):
       if len(config[1]) <= 1:
          return False

       for i in range(config[0], len(config[2])):
          if gold_heads[i] == config[1][-1]:
             return False
       return True
    
    @staticmethod
    def zero_cost_left(config, gold_heads):
       if (config[0] >= len(config[2])) or (len(config[1]) < 1):
          return False
       
       for i in config[1]:
          if config[0] == gold_heads[i]:
             return False
       return True
  
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
      predict = parser.predict([t[0] for t in data], [t[1] for t in data], parser)
      gold = [t[2] for t in data]
      correct += sum(a == b for a, b in zip(predict[1:len(predict)], gold[1:len(gold)]))
      total += len(predict)-1
    return correct/total

def oracle_moves_dynamic(gold_heads, parser, config):

    #config = parser.initial_config(len(gold_heads))

    #for i in range(2*len(gold_heads)-1):
    valid_moves = parser.valid_moves(config)
    options = []
    if (parser.zero_cost_shift(config, gold_heads)) and (0 in valid_moves):
      options.append(0)
    if (parser.zero_cost_left(config, gold_heads)) and (1 in valid_moves):
      options.append(1)
    if (parser.zero_cost_right(config, gold_heads)) and (2 in valid_moves):
      options.append(2)
    if len(options) == 0:
      moves = sample(valid_moves,1)
      return moves
    else:
      return options


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
    if gold_heads != config[2]:
       print('ERROR!!!!')

          

def training_examples(vocab_words, vocab_tags, gold_data, parser, batch_size=100):
    batch = torch.zeros((batch_size, 4, 200))
    moves = torch.zeros(batch_size)
    index = 0
    not_final = False
    for data in gold_data:
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
        if not_final:
           rnn_embeddings = parser.lstm_model.forward(words_id, tags_id)
           not_final = False
        row = parser.featurize(rnn_embeddings, config)
        moves[index] = move
        batch[index] = row
        if index == (batch_size - 1):
          yield batch, moves
          if not parser.is_final_config(config):
             not_final = True
          index = -1
          batch = torch.zeros((batch_size, 4, 200))
          moves = torch.zeros(batch_size)
        index += 1

    if index < batch_size:
      batch = batch[0:index]
      moves = moves[0:index]
      yield batch, moves

def batchify(vocab_words, vocab_tags, gold_data, parser, batch_size=100):
    batch = []
    moves = []
    index = 0
    for data in gold_data:
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
        moves.append(move)
        batch.append(row)
      yield torch.stack(batch), torch.tensor(moves)
      batch = []
      moves = []

def train_fixed_window(train_data,vocab_words, vocab_tags, n_epochs=1, batch_size=100, lr=1e-2):
    print('Started training parser...')
     # Initialize the model and tagger
    parser = FixedWindowParser(vocab_words, vocab_tags)

    # Initialize the optimizer. Here we use Adam rather than plain SGD
    optimizer = optim.Adam(parser.model.parameters(), lr=lr)
    start2 = time.time()
    for epoch in range(n_epochs):
        print('Epoch', (epoch+1), '/', n_epochs)
        #parser.model.train()
        losses = []
        batch_test = torch.rand((100,4,200))
        output_test = torch.rand(3)
        with tqdm(total=421700) as pbar:
          for batch, moves in training_examples(vocab_words, vocab_tags, train_data, parser, batch_size):
             # print('Training examples:', time.time()-start2)
              start1 = time.time()
              optimizer.zero_grad()
              #batch = batch.detach()
              output = parser.model.forward(batch)
              output = output.squeeze(1)
              moves = moves.long()
              loss = F.cross_entropy(output, moves)
              loss.backward()
              optimizer.step()
              losses.append(loss.item())
              pbar.set_postfix(loss = (sum(losses)/len(losses)))
              pbar.update(batch_size)
              start2 = time.time()
             # print('Loss:', time.time()-start1)
    return parser

def run_parser(vocab_words, vocab_tags, train_data, dev_data):
   # Train the parser
   trained_parser = train_fixed_window(train_data, vocab_words, vocab_tags)

   # Calculate accuracy
   acc = uas(trained_parser, dev_data)

   return acc
