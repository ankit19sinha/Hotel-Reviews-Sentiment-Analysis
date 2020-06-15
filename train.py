# For pretty print of tensors.
# Must be located at the first line except the comments.
from __future__ import print_function

# Import the basic modules.
import random
import argparse
import numpy as np

# Import the PyTorch modules.
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torchtext import data

# Import all the models
import models.vRNN 
import models.GRU
import models.LSTM
import models.attn_RNN

# Initilize a command-line option parser.
parser = argparse.ArgumentParser(description='Sentiment Analysis')

# Add a list of command-line options that users can specify.
# Shall scripts (.sh) files for specifying the proper options are provided. 
parser.add_argument('--lr', type=float, metavar='LR', help='learning rate')
parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', help='Input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', help='Number of epochs to train')
parser.add_argument('--model', choices=['vRNN', 'GRU', 'LSTM', 'attention'], help='which model to train/evaluate')
parser.add_argument('--hidden_dim', type=int, help='number of hidden features')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability to be applied')
parser.add_argument('--num_layers', type=int, metavar='N', help='Number of RNN layers')
parser.add_argument('--useGlove', type=bool, help='Flag for using GloVe Representation')
parser.add_argument('--trainable', type=bool, help='Set requires_grad=False for GloVe representation matrix')
parser.add_argument('--bidirectional', type=bool, help='Flag to determine uni-directional or bidirectional RNN')
parser.add_argument('--typeOfPadding', choices=['pack_padded', 'padded', 'no_padding'], help='which type of padding to use')
parser.add_argument('--typeOfRNN', choices=['simple', 'GRU', 'LSTM'], help='which type of RNN to use with attention')
parser.add_argument('--typeOfAttention', choices=['multiplicative', 'additive'], help='which type of attention to use')

# Add more command-line options for other configurations.
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='number of batches between logging train status')

# Parse the command-line option.
args = parser.parse_args()

# CUDA will be supported only when user wants and the machine has GPU devices.
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Change the random seed.
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Set the device-specific arguments if CUDA is available.
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# Define fields for the data
TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True, lower=True)
LABEL = data.LabelField(dtype = torch.float, batch_first=True)
fields = [(None, None), ('text',TEXT),('label', LABEL)]

# Import preprocessed dataset
reviews_data = data.TabularDataset(path = 'hotel_reviews_processed.csv',
                                   format = 'csv',
                                   fields = fields,
                                   skip_header = True)

# Split training and testing data
train_data, test_data = reviews_data.split(split_ratio=0.75, random_state = random.seed(args.seed))

# Build the vocabulary
TEXT.build_vocab(train_data, min_freq = 3, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# Save the glove embedding in a matrix
embedding_matrix = TEXT.vocab.vectors

# Check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

# Make iterator for splits
train_iter, test_iter = data.BucketIterator.splits(
    (train_data, test_data), batch_size=args.batch_size, device=device,
    sort_key = lambda x: len(x.text),
    sort_within_batch=True)

# Initiate models according to the argument
if args.model == 'vRNN':
    model = models.vRNN.vRNNModel(embedding_matrix, args.hidden_dim, args.num_layers, args.dropout, args.bidirectional, args.useGlove, args.trainable, args.typeOfPadding)
elif args.model == 'GRU':
    model = models.GRU.gruModel(embedding_matrix, args.hidden_dim, args.num_layers, args.dropout, args.bidirectional, args.useGlove, args.trainable, args.typeOfPadding)
elif args.model == 'LSTM':
    model = models.LSTM.lstmModel(embedding_matrix, args.hidden_dim, args.num_layers, args.dropout, args.bidirectional, args.useGlove, args.trainable, args.typeOfPadding)
elif args.model == 'attention':
    model = models.attn_RNN.attn_RNNModel(embedding_matrix, args.hidden_dim, args.num_layers, args.dropout, args.bidirectional, args.useGlove, args.trainable, args.typeOfPadding, args.typeOfRNN, args.typeOfAttention)
else:
    raise Exception('Unknown model {}'.format(args.model))

# Activate CUDA if specified and available.
if args.cuda:
    model.cuda()

# Use BinaryCrossEntropyLoss  
criterion = nn.BCELoss()
criterion = criterion.to(device)

# Use Adam for optimization
optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)


# Function: train the model just one iteration.
def train(epoch):
    # For each batch of reviews,
    for batch_idx, batch in enumerate(train_iter):
        
        # Put model in training mode
        model.train()
    
        # Reset the gradients
        optimizer.zero_grad()   
        
        # Retrieve text and number of words
        text, text_lengths = batch.text   
        
        # Convert predictions into a 1D tensor
        predictions = model(text, text_lengths).squeeze()

        # Calculate loss
        loss = criterion(predictions, batch.label)        
        
        # Backprop
        loss.backward()       
        
        # Update the weights
        optimizer.step()
        
        # Print out the loss and accuracy on the first 4 batches of the testing dataset.
        if batch_idx % args.log_interval == 0:
            # Compute the average testing loss and accuracy.
            test_loss, test_acc = evaluate(n_batches=4)
            
            # Compute the training loss.
            train_loss = loss.data.item()
            
            # Compute the number of examples in this batch.
            examples_this_epoch = batch_idx * len(batch)

            # Compute the progress rate in terms of the batch.
            epoch_progress = 100. * batch_idx / len(train_iter)

            # Print out the training loss, testing loss, and accuracy with epoch information.
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                  'Train Loss: {:.6f}\tTesting Loss: {:.6f}\tTesting Acc: {}'.format(
                epoch, examples_this_epoch, len(train_iter.dataset),
                epoch_progress, train_loss, test_loss, test_acc))

# Function: evaluate the learned model on test data.
def evaluate(verbose=False, n_batches=None):
    
    #Put model in evaluation mode
    model.eval()

    # Initialize cumulative loss and the number of correctly predicted examples.  
    loss = 0
    correct = 0
    cumsum_correct = 0
    n_examples = 0

    # For each batch in the loaded dataset,
    with torch.no_grad():
        for batch_i, batch in enumerate(test_iter):        
            # Retrieve text and no. of words
            text, text_lengths = batch.text
            
            # Convert predictions into a 1D tensor
            predictions = model(text, text_lengths).squeeze()
            
            # Accumulate the loss by comparing the predicted output and the true target labels.
            loss += criterion(predictions, batch.label).data
            
            # Round up the predictions to their nearest integer
            rounded_preds = torch.round(predictions)
            
            # Number of correct predictions
            correct = (rounded_preds == batch.label).float()
            
            # Calculate cumulative correct predictions
            cumsum_correct += correct.sum()
    
            # Keep track of the total number of predictions.
            n_examples += rounded_preds.size(0)
    
            # Skip the rest of evaluation if the number of batches exceed the n_batches.
            if n_batches and (batch_i >= n_batches):
                break
    
    # Compute the average loss per example.
    loss /= n_examples

    # Compute the average accuracy in terms of percentile.
    acc = 100. * (cumsum_correct / n_examples)

    # If verbose is True, then print out the average loss and accuracy.
    if verbose:        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss, cumsum_correct, n_examples, acc))
    return loss, acc


## Train the model one epoch at a time.
for epoch in range(1, args.epochs + 1):
    train(epoch)

## Evaluate the model on the test data and print out the average loss and accuracy.
## Note that you should use every batch for evaluating on test data rather than just the first four batches.
evaluate(verbose=True)

## Save the model (architecture and weights)
torch.save(model, args.model + '.pt')
