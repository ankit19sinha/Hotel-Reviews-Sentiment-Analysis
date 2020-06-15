import torch
import torch.nn as nn
from torch.nn import functional as F

class attn_RNNModel(nn.Module):
  def __init__(self, embedding_matrix=None, hidden_dim=None, num_layers=None, dropout=None, bidirectional=None, useGlove=None, trainable=None, typeOfPadding="no_padding", typeOfRNN="simple", typeOfAttention="multiplicative"):
    super().__init__()
    # Create an embedding layer of dimension vocabulary size * 100
    self.embeddingLayer = nn.Embedding(len(embedding_matrix), 100)
    
    # Set embedding to GloVe representation if flag is True
    if useGlove:
        self.embeddingLayer.weight.data.copy_(embedding_matrix)
        
    # Set weights of the matrix to non-updatable if flag is False 
    if not trainable:
        self.embeddingLayer.weight.requires_grad = False
    
    # Initialize the RNN structure
    self.typeOfRNN = typeOfRNN
    
    if typeOfRNN == "simple":
        self.rnnLayer = nn.RNN(input_size = 100, 
                           hidden_size = hidden_dim,
                           batch_first = True,
                           num_layers = num_layers,
                           dropout = dropout,
                           bidirectional = bidirectional)
    elif typeOfRNN == "GRU":
        self.rnnLayer = nn.GRU(input_size = 100, 
                           hidden_size = hidden_dim,
                           batch_first = True,
                           num_layers = num_layers,
                           dropout = dropout,
                           bidirectional = bidirectional)
    else:
        self.rnnLayer = nn.LSTM(input_size = 100, 
                           hidden_size = hidden_dim,
                           batch_first = True,
                           num_layers = num_layers,
                           dropout = dropout,
                           bidirectional = bidirectional)
    
    # Initialize variables to define the fully-connected layer
    self.num_directions = 2 if bidirectional else 1
    self.typeOfPadding = typeOfPadding
    self.typeOfAttention = typeOfAttention
    self.num_layers = num_layers
    self.hidden_size = hidden_dim
    
    # Initialize the learnable attention weights
    if self.typeOfAttention == "multiplicative":
        self.attn_weight = nn.Linear(hidden_dim * self.num_directions, hidden_dim * self.num_directions)
    else:
        self.Wh = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size * self.num_directions)
        self.Ws = nn.Linear(self.hidden_size * self.num_directions, self.hidden_size * self.num_directions)

    # Initialize a fully-connected layer
    self.fc = nn.Linear(hidden_dim * self.num_directions, 1)
    
    # Sigmoid activation function squashes the output between 0 and 1
    self.sigmoid = nn.Sigmoid()

  def forward(self, text, text_lengths):
    # For given texts, get their embeddings
    embedded = self.embeddingLayer(text)
    
    # Pad the embedding if typeOfPadding parameter has been set
    if self.typeOfPadding == "padded":
        # Create a padded sequence to handle variable length inputs
        embedded = nn.utils.rnn.pad_sequence(
            embedded,
            batch_first=True)
        
    # Pass the embedded tensor with or without padding to the RNN
    # Get the outputs and hidden state from the end of the RNN
    if self.typeOfRNN == "LSTM":
        output, (hidden_n, cell_n) = self.rnnLayer(embedded)
    else:
        output, hidden_n = self.rnnLayer(embedded)
    
    # Apply attention mechanism
    attn = self.attention(output, hidden_n)
    
    # Pass the obtained weighted encoder state to the fully-connected layer
    output = self.fc(attn)
      
    # Squash the output between 0 and 1
    output = self.sigmoid(output)
    
    return output

  def attention(self, output, hidden_n):
    # output has the following dimension: [batch, seq_len, num_directions * hidden_size]
    # And it contains all output states
    # Final hidden state is of [batch, num_layers * num_directions, hidden_size]
    
    # Rearrange hidden state to [num_layers, batch, num_directions * hidden_size]
    hidden_n = hidden_n.view(self.num_layers, -1, self.num_directions * self.hidden_size)
    
    if self.typeOfAttention == "multiplicative":
        # Take the hidden state only from the last layer
        # s = hidden_n = [batch, direction * hidden]
        # W = [direction * hidden, direction * hidden]
        # x = [batch, direction * hidden]
        hidden_n = hidden_n[-1::].squeeze()
    
        # Pass the hidden state to the attention weight layer
        x = self.attn_weight(hidden_n)
    
        # Batch mulitply output and attention's hidden state
        # output = [batch, seq_len, direction * hidden]
        # x = [batch, direction * hidden, 1]
        # attn_scores has the dimension of [batch, seq_len, 1]
        attn_scores = torch.bmm(output, x.unsqueeze(-1))
    
    else:
        # Initialize attention parameter
        self.attn_v = nn.Parameter(torch.FloatTensor(output.shape[0], self.num_directions * self.hidden_size)).cuda()
        
        # Rearrange hidden_n to [batch, 1, direction * hidden]
        hidden_n = hidden_n[-1::].permute(1, 0, 2)

        # Additive attention = v * tanh(Wh * output + Ws * hidden_n)
        # Compute product of Wh and outputs from RNN
        x = self.Wh(output)
        
        # Compute product of Ws and final hidden state
        y = self.Ws(hidden_n)
        
        # z has the dimension of [batch, seq_len,  direction * hidden]
        z = torch.tanh(x + y)

        # Batch mulitply z and attention parameter v
        # attn_scores has the dimension of [batch, seq_len, 1]
        attn_scores = torch.bmm(z, self.attn_v.unsqueeze(-1))
    
    
    # Apply softmax on seq_len to get attention distribution
    attn_distb = F.softmax(attn_scores, 1)
    
    # Batch multiply output and attention distribution 
    # weighted_encoder_state has the following dimension: [batch, num_directions * hidden_size]
    weighted_encoder_state = torch.bmm(output.permute(0, 2, 1), attn_distb).squeeze(-1)
    
    return weighted_encoder_state
