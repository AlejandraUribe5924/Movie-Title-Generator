## Movie-Title-Generator

The following project is a Recurrent Neural Network (RNN) trained on 17,062 American movie titles. Using Byte-Pair (BPE) Tokenization, a new vocabulary is created based on recurring patterns which are recorded as tokens.

### tokenizer.py
The Basic Tokenizer class in which the vocabulary is made based on the vocab_size hyperparameter and sequence_length. 

### layers.py
The layers of the model which include the Embedding, LSTMCell (Long-short Term Memory), LSTM, and Linear.

### model.py
The MovieTitleGenerator model which includes the forward pass and the parameters.

### load_dataset.py
The accessing of the movie titles and where the training (90% of tokens) and validation (10% of tokens) tensors are created.

### title_generator.ipynb
Where the model training is done. This file is currently ongoing and will continue to see changes. 

## Contributions and References

I used Andrej Kaparthy's [Tokenization tutorial video](https://www.youtube.com/watch?v=zduSFxRajkE&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=9&ab_channel=AndrejKarpathy) and the accompaning [GitHub respository](https://github.com/karpathy/minbpe/tree/master) to build the basic tokenizer. 

I also used Paul Sun's [nameWithCty repository](https://github.com/huiprobable/nameWithCty/tree/main) to get a better understanding of what is required for my own neural net (i.e. necessary files and how they relate to the neural net training). I used a combination of the two repositories to create 'model.py', 'layers.py', and 'load_dataset.py'.

I used the following resources to help better understand the workings of the LSTM cell : Neural networks - why is there Tanh(x)*sigmoid(x) in a LSTM cell? - artificial intelligence stack exchange. AI Stack Exchange. (2021, November 24). https://ai.stackexchange.com/questions/32505/why-is-theretanhxsigmoidx-in-a-lstm-cell

