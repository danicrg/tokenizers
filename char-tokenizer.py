class Tokenizer:
    """
    The simplest character based tokenizer
    """
    def __init__(self):
        self.vocab = []
        self.vocab_size = 0

    def train(self, text):
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)

        self.stoi = {ch:i for i, ch in enumerate(self.vocab)}
        self.itos = {i:s for i, ch in enumerate(self.vocab)}

    def encode(self, text):
        return [self.stoi.get(char, "-1") for char in text]

    def decode(self, ints):
        return [self.itos.get(i, "UNK") for i in ints]