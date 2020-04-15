import torch
import torch.nn as nn

class vRNNModel(nn.Module):
  def __init__(self, embedding_matrix=None, hidden_dim=None, num_layers=None, dropout=None, bidirectional=None, useGlove=None, trainable=None, typeOfPadding="no_padding"):
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
    self.rnnLayer = nn.RNN(input_size = 100, 
                           hidden_size = hidden_dim, 
                           num_layers = num_layers, 
                           batch_first = True,
                           bidirectional = bidirectional,
                           dropout = dropout)
    
    # Initialize variables to define the fully-connected layer
    self.num_directions = 2 if bidirectional else 1
    self.hidden_size = hidden_dim
    self.num_layers = num_layers
    self.typeOfPadding = typeOfPadding

    # Initialize a fully-connected layer
    self.fc = nn.Linear(self.hidden_size * self.num_directions, 1)
    
    # Sigmoid activation function squashes the output between 0 and 1
    self.sigmoid = nn.Sigmoid()

  def forward(self, text, text_lengths):
    # For given texts, get their embeddings
    embedded = self.embeddingLayer(text)
    
    if self.typeOfPadding == "pack_padded":
        # Create packed padded sequence to handle variable length inputs
        embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            text_lengths,
            batch_first=True)
    
    elif self.typeOfPadding == "padded":
        # Create a padded sequence to handle variable length inputs
        embedded = nn.utils.rnn.pad_sequence(
            embedded,
            batch_first=True)
        
    else:
        # Do nothing and leave embedding as it is
        pass
    
    # Get the outputs and hidden state from the end of the RNN
    output, hidden = self.rnnLayer(embedded)
    
    # Rearrange hidden_n to [num_layers, batch, hidden_size * num_directions]
    hidden = hidden.view(self.num_layers, -1, self.hidden_size * self.num_directions)
    
    # Pick the hidden states of the batch at the last layer and pass to fc
    output = self.fc(hidden[-1::].squeeze())
  
    # Squash the output between 0 and 1
    output = self.sigmoid(output)
    
    return output
