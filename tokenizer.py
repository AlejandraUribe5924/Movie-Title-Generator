## Counting the number of times each pair of consecutive ids appears in a list
# input: list of integer ids
# returns: dictionary with counts of each pair of consecutive ids
def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

# ------------------------------------------------------------------------------------

## Merging the most common pair of tokens and making them a new token
# input: (ids) list of integers, (pair) the common pair of tokens, (idx) the new token index
# returns: a new list of integers with the common pair replaced with the new token
def merge(ids, pair, idx):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

# -----------------------------------------------------------------------------------

class BasicTokenizer:

    def __init__(self):
        self.merges = {}
        self.patterns = ""
        self.special_tokens = {}
        #self.vocab = self._build_vocab() 

    def train(self, text, vocab_size, verbose = False):
        self.merges = {}
        self.vocab = {}
        #vocab size is a hyperparameter that will be tuned later
        num_merges = vocab_size - 256

        # input text processing
        text_bytes = text.encode('utf-8') # This is where the text is converted to bytes
        ids = list(text_bytes) # Making a copy of the tokens (text_bytes) list

        ## Making new tokens by merging the most common pairs of tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats: 
                print("No more pairs to merge.")
                break
            pair = max(stats, key = stats.get)
            idx = 256 + i
            #print(f"merging {pair} into a new token {idx}")
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f"merging {pair} into a new token {idx}")

        self.merges = merges # Saving to use in encode()
        self.vocab = vocab # Saving to use in decode()
        print(f"Number of merges: {len(merges)}")
    
    def encode(self, text):
        """Encode a string into a list of integer tokens."""
        tokens = list(text.encode("utf-8")) # raw bytes
        # We need to merge from top to bottom since some of the later merges depend on the earlier ones
        while len(tokens) >=2: # We want to repeatedly merge pairs of tokens as long as valid merges exist in the dictionary
            stats = get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) # returns the most legible candidate pair that occurs in the tokens
            # this may not work if there is nothing to get from the merges. It will just the first one and the rest will be float("inf")
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx) # replace the original tokens replacing the pair with the idx. Every occurence of pair is changed to idx
        return tokens
    
    def decode(self, ids):
        """Decode a list of integers into a string."""
        tokens = b"".join(self.vocab[idx] for idx in ids) # joining the byte objects into a single byte string
        text = tokens.decode("utf-8", errors="replace") # decoding them into a string
        return text

  
    def _build_vocab(self):
        """Build a vocabulary from the text."""
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1] # Addition of byte objects
        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode("utf-8")
        return vocab
    