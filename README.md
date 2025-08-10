# CS336 Spring 2025 Assignment 1: BPE Implementation from Scratch

(Ongoing)


Personal implementation/solution of a BPE from scratch, following the assignment handout at cs336_spring2025_assignment1_basics.pdf

BPE (Byte-Pair Encoding) is a popular tokenizer adapted to word tokenization by "Neural Machine Translation of Rare Words with Subword Units". It is a data compression technique that iteratively replaces the most frequent pair of characters in a sequence with a single, unused byte. 

The following implementation is trained on TinyStories data.

Comments useful to comprehension were voluntarily kept in the repo.

### Download the TinyStories data and a subsample of OpenWebText

```
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
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
