import torch
import torch.nn as nn

class gruModel(nn.Module):
  def __init__(self, embedding_matrix, hidden_dim, num_layers, dropout, bidirectional, useGlove, trainable):
    super().__init__()
    # Create an embedding layer of dimension vocabulary size * 100
    self.embeddingLayer = nn.Embedding(len(embedding_matrix), 100)
    
    # Set embedding to GloVe representation if flag is True
    if useGlove:
        self.embeddingLayer.weight.data.copy_(embedding_matrix)
        
    # Set weights of the matrix to non-updatable if flag is False 
    if not trainable:
        self.embeddingLayer.weight.requires_grad = False
        
    # Initialize GRU architecture
    self.gruLayer = nn.GRU(input_size = 100, 
                           hidden_size = hidden_dim, 
                           num_layers = num_layers, 
                           batch_first = True,
                           bidirectional = bidirectional,
                           dropout = dropout)
    
    # Initialize a fully-connected layer
    self.fc = nn.Linear(hidden_dim*2, 1)
    
    # Sigmoid activation function squashes the output between 0 and 1
    self.sigmoid = nn.Sigmoid()

  def forward(self, text, text_lengths):
    # For given texts, get their embeddings
    embedded = self.embeddingLayer(text)
    
    # Create padded sequence so that inputs of variable length can be handled
    packed_embedded = nn.utils.rnn.pack_padded_sequence(
        embedded, 
        text_lengths,
        batch_first=True)
    
    # Get the output and hidden state from the end of the GRU
    output, hidden_n = self.gruLayer(packed_embedded)
    
    # Concatenate the final forward and backward hidden states
    hidden = torch.cat((hidden_n[-2,:,:], hidden_n[-1,:,:]), dim = 1)

    # Pass through the full-connected layer
    output = self.fc(hidden)
    
    # Squash the output between 0 and 1
    output = self.sigmoid(output)
    
    return output
