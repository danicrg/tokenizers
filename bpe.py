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
        
    def merge(self, ids, pair, idx):
        """
        Replaces the `pair` in the array `ids` with the `idx`
        """
        new_ids = []
        i = 0
        while i < len(ids):

            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1

        return new_ids


def main():
    """
    Testing out the get_stats function
    """

    text = "The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to strings. In this lecture we build from scratch the Tokenizer used in the GPT series from OpenAI. In the process, we will see that a lot of weird behaviors and problems of LLMs actually trace back to tokenization. We'll go through a number of these issues, discuss why tokenization is at fault, and why someone out there ideally finds a way to delete this stage entirely."
    ids = text.encode()
    stats = BPETokenizer().get_stats(ids)

    for count, pair in sorted(((v, k) for k, v in stats.items()), reverse=True)[:10]:
        print(count, chr(pair[0]), chr(pair[1]))

    top_pair = max(stats, key=stats.get)

    new_ids = BPETokenizer().merge(ids, top_pair, 256)
    print(f"{len(ids)=}")
    print(f"{len(new_ids)=}")



if __name__ == "__main__":
    main()