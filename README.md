# Tokenizers
This is a repo for implementing tokenizers for fun.

---

Tokenization is at the heart of much weirdness of LLMs. 
- LLMs struggle to spell words
- LLMs struggle to reverse a string
- LLM is bad at simple arithmetic

Cool [link](tiktokenizer.vercel.app) to visualize different tokenizers.


## Character level encoding

With this tokenizer, each character in the text will correspond to an integer. The training is incredibly simple: go through each character in the training text and assign an integer.

**Advantages**:
- Very small vocabulary size, which leads to small embedding tables in transformers.
- Fast to train. Requires one run through the text.

**Disadvantages**:
- There is no compression of the length of the sequence, which reduces the context lenght in a transformer and increases the complexity.

## Byte pair encoding

Algorithm: iteratively replace the most common contiguous sequences of characters in a target piece of text with unused 'placeholder' bytes.

## Considerations
### Vocabulary size

The number of parameters on a transformer model is dependent on the vocabulary size. So higher vocab_size will mean more parameters. Specifically, in the embedding table and the head linear layer at the end of the decoder.
