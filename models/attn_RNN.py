import torch
import torch.nn as nn
from torch.nn import functional as F

class attn_RNNModel(nn.Module):
  def __init__(self, embedding_matrix, hidden_dim, bidirectional, useGlove, trainable, typeOfPadding="no_padding", typeOfRNN="simple"):
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
                           bidirectional = bidirectional)
    elif typeOfRNN == "GRU":
        self.rnnLayer = nn.GRU(input_size = 100, 
                           hidden_size = hidden_dim,
                           batch_first = True,
                           bidirectional = bidirectional)
    else:
        self.rnnLayer = nn.LSTM(input_size = 100, 
                           hidden_size = hidden_dim,
                           batch_first = True,
                           bidirectional = bidirectional)
    
    # Initialize variables to define the fully-connected layer
    self.num_directions = 2 if bidirectional else 1
    self.typeOfPadding = typeOfPadding
    self.hidden_size = hidden_dim
    
    # Initialize the learnable attention weight
    self.attn_weight = nn.Linear(hidden_dim * self.num_directions, hidden_dim * self.num_directions)

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
    # And it contains all hidden states
    # Final hidden state is of [batch, 1 * num_directions, hidden_size]
    # Rearrange hidden state to [batch, num_directions * hidden_size, 1]
    hidden_n = hidden_n.view(-1, self.num_directions * self.hidden_size)

    # Pass the hidden state to the attention weight layer
    x = self.attn_weight(hidden_n)

    # Batch mulitply output and attention's hidden state
    # attn_scores has the dimension of [batch, seq_len, 1]
    attn_scores = torch.bmm(output, x.unsqueeze(-1))
    
    # Apply softmax on seq_len to get attention distribution
    attn_distb = F.softmax(attn_scores, 1)
    
    # Batch multiply output and attention distribution 
    # weighted_encoder_state has the following dimension: [batch, num_directions * hidden_size]
    weighted_encoder_state = torch.bmm(output.permute(0, 2, 1), attn_distb).squeeze(-1)
    
    return weighted_encoder_state