class BPETokenizer:
    def __init__(self, vocab_size=276):
        self.vocab_size = vocab_size

    def get_stats(self, ids):
        """
        Returns the frequencies of each pair found in the list of ids
        """

        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """
        Replaces the `pair` in the array `ids` with the `idx`
        """
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids

    def train(self, text):
        ids = text.encode()
        n_iterations = self.vocab_size - 256
        self.merged = {}
        new_ids = list(ids)
        for i in range(n_iterations):
            stats = self.get_stats(new_ids)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            new_ids = self.merge(new_ids, top_pair, idx)
            self.merged[top_pair] = idx

        print("Compression ratio:", len(ids) / len(new_ids))

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merged.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        return new_ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        tokens = text.encode()

        while True:
            stats = self.get_stats(tokens)

            # Get the lowest pair in the merged dictionary
            pair = min(stats, key=lambda p: self.merged.get(p, float("inf")))

            if pair not in self.merged:
                break

            idx = self.merged[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens


def main():
    """
    Testing out the encode function
    """

    text = "The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to strings. In this lecture we build from scratch the Tokenizer used in the GPT series from OpenAI. In the process, we will see that a lot of weird behaviors and problems of LLMs actually trace back to tokenization. We'll go through a number of these issues, discuss why tokenization is at fault, and why someone out there ideally finds a way to delete this stage entirely."
    tokenizer = BPETokenizer()
    new_ids = tokenizer.train(text)
    enc_ids = tokenizer.encode(text)
    assert new_ids == enc_ids


if __name__ == "__main__":
    main()
