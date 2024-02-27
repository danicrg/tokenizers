class BPETokenizer:
    def __init__(self):
        pass

    def get_stats(self, ids):
        """
        Returns the frequencies of each pair found in the list of ids
        """

        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
        

def main():
    """
    Testing out the get_stats function
    """

    text = "The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to strings. In this lecture we build from scratch the Tokenizer used in the GPT series from OpenAI. In the process, we will see that a lot of weird behaviors and problems of LLMs actually trace back to tokenization. We'll go through a number of these issues, discuss why tokenization is at fault, and why someone out there ideally finds a way to delete this stage entirely."
    ids = text.encode()
    print(sorted(BPETokenizer().get_stats(ids), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    main()