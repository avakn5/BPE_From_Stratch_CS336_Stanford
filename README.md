# CS336 Spring 2025 Assignment 1: BPE Implementation from Scratch

This repository contains my implementation of Byte Pair Encoding (BPE) from scratch for CS336 — Spring 2025 Assignment 1, based on the provided assignment handout (cs336_spring2025_assignment1_basics.pdf).

Byte Pair Encoding is a subword tokenization algorithm originally adapted for NLP by Sennrich et al., Neural Machine Translation of Rare Words with Subword Units (2016). It works by iteratively replacing the most frequent pair of symbols in the text with a new symbol, effectively building a vocabulary that balances character-level and word-level representations.

__The codebase includes extensive inline comments to aid understanding of the BPE implementation.__

![Transformer Architecture](/Users/akouhana/CS336/assignment1-basics/cs336_basics/figure/architecture.jpeg)

### Download the TinyStories data and a subsample of OpenWebText

```
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```

### STEPS to the BPE implementation: 

* 1- Pre-process the text.
* 2- Pretokenize the text.
* 3- UTF8 encode each pretoken.
* ------- Merging --------------
* 4- Split each pretoken into bytes.
* 5- count pair frequencies.
* 6- merge most frequent pairs.
* 7- repeat from step 4 until reached the vocab_size.

### Implemented Features:

* BPE training procedure.
* Encoder (text → token IDs).
* Decoder (token IDs → text).
* Passing all assignment tests:
&nbsp;BPE training: 3/3 tests passed
&nbsp; Tokenizer: 23/23 tests passed
