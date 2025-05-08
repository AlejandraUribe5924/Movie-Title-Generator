import pandas as pd
import csv
from tokenizer import BasicTokenizer
import torch
from layers import Embedding, LSTM, Linear

## Read the CSV file using pandas
df = pd.read_csv('movies.csv', engine='python', quotechar='"', on_bad_lines='skip')

## Creating a list of movie titles from the CSV file
movies = []
with open('movies.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        rowlist = row[0].split(",")
        movie = rowlist[0]
        if movie != '':
            movies.append(movie)

## Joining the list of movie titles into a single string
all_movies = ' '.join(movies)

def load_dataset(all_movies, vocab_size = 276, seq_len = 20):

    ## Training the tokenizer
    tokenizer = BasicTokenizer()
    tokenizer.train(all_movies, vocab_size = vocab_size, verbose = False)
    ## Encoding into a long sequence of tokens
    tokens = tokenizer.encode(all_movies)

    def make_sequences(tokens, seq_len = 20):
        X, Y = [], []
        for i in range(len(tokens) - seq_len):
            X.append(tokens[i:i + seq_len])
            Y.append(tokens[i + seq_len])

        X = torch.tensor(X, dtype=torch.long)
        Y = torch.tensor(Y, dtype=torch.long)

        return X, Y

    ## Splitting the dataset into training (90%) and validation (10%)
    ## We must keep the order of the tokens (ie. no shuffling) in order to preserve the sequence information
    split_idx = int(0.9 * len(tokens))
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    ## Creating the training and validation torch tensors
    X_train, Y_train = make_sequences(train_tokens, seq_len = seq_len)
    X_val, Y_val = make_sequences(val_tokens, seq_len = seq_len)

    return X_train, Y_train, X_val, Y_val, tokenizer
