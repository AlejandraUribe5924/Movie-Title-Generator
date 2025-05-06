# Movie-Title-Generator

The following project is a Recurrent Neural Network (RNN) trained on 17,062 American movie titles. Using Byte-Pair (BPE) Tokenization, a new vocabulary is created based on recurring patterns which are recorded as tokens.

## tokenizer.py
The Basic Tokenizer class in which the vocabulary is made based on the vocab_size hyperparameter and sequence_length. 

## layers.py
The layers of the model which include the Embedding, LSTMCell (Long-short Term Memory), LSTM, and Linear.

## model.py
The MovieTitleGenerator model which includes the forward pass and the parameters.

## load_dataset.py
The accessing of the movie titles and where the training (90% of tokens) and validation (10% of tokens) tensors are created.
