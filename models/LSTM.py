import torch
import torch.nn as nn

class lstmModel(nn.Module):
  def __init__(self, embedding_matrix, hidden_dim, num_layers, dropout, bidirectional, useGlove, trainable, typeOfPadding="no_padding"):
    super().__init__()
    # Create an embedding layer of dimension vocabulary size * 100
    self.embeddingLayer = nn.Embedding(len(embedding_matrix), 100)
    
    # Set embedding to GloVe representation if flag is True
    if useGlove:
        self.embeddingLayer.weight.data.copy_(embedding_matrix)
        
    # Set weights of the matrix to non-updatable if flag is False 
    if not trainable:
        self.embeddingLayer.weight.requires_grad = False
        
    # Initialize LSTM architecture
    self.lstmLayer = nn.LSTM(
        input_size = 100,
        hidden_size = hidden_dim,
        num_layers = num_layers,
        batch_first = True,
        dropout = dropout,
        bidirectional = bidirectional
    )
    
    # Initialize variables to define the fully-connected layer
    num_directions = 2 if self.lstmLayer.bidirectional else 1
    self.typeOfPadding = typeOfPadding

    # Initialize a fully-connected layer
    if typeOfPadding == "pack_padded":
      self.fc = nn.Linear(hidden_dim * 2, 1)
    else:
      self.fc = nn.Linear(hidden_dim * num_directions, 1)
    
    # Sigmoid activation function squashes the output between 0 and 1
    self.sigmoid = nn.Sigmoid()

  def forward(self, text, text_lengths):
    # For given texts, get their embeddings
    embedded = self.embeddingLayer(text)
    
    if self.typeOfPadding == "pack_padded":
        # Create padded sequence so that inputs of variable length can be handled
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            text_lengths,
            batch_first=True)
        
        # Get the output, hidden state, and cell state from the end of the LSTM layer
        output, (hidden_n, cell_n) = self.lstmLayer(packed_embedded)
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden_n[-2,:,:], hidden_n[-1,:,:]), dim = 1)
        
        # Pass through the full-connected layer 
        output = self.fc(hidden)
        
    elif self.typeOfPadding == "padded":
        # Create a padded sequence to handle variable length inputs
        padded_embedded = nn.utils.rnn.pad_sequence(
            embedded,
            batch_first=False)
            
        # Get the output and the hidden state from the end of the LSTM
        output, (hidden_n, cell_n) = self.lstmLayer(padded_embedded)
        
        # Pass the final output state through the fully-connected layer
        # Output is in the format of [seq_len, batch, num_directions * hidden_size]
        # We pick the last index of the seq_len dimension and squeeze the tensor
        # New dimension of output becomes [batch, num_directions * hidden_size]
        output = self.fc(output[-1::].squeeze())
        
    else:
        # Pass the embedded tensor without padding
        # Get the output and hidden state from the end of the LSTM
        output, (hidden_n, cell_n) = self.lstmLayer(embedded)
        
        # Rearrange output to [seq_len, batch, num_directions * hidden_size]
        output = output.permute(1, 0, 2)
        
        # Similar to padded sequence, output needs to in the format of 
        # [batch, num_directions * hidden_size]
        output = self.fc(output[-1::].squeeze())
    
    # Squash the output between 0 and 1
    output = self.sigmoid(output)
    
    return output
