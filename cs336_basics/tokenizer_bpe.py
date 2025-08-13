import regex as re
from collections import defaultdict
import cProfile, pstats
from typing import Iterable, Iterator
from cs336_basics.train_bpe import BPE
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #Pre-tokenization pattern
PAT_RE = re.compile(PAT)

class BPE_Tokenizer(): 
    '''Construct a Tokenizer from a given vocabulary, list of merges, and a list of special tokens 
    and uses them to encode and decode text to/from token IDs.'''
     
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None): 
        self.special_tokens = special_tokens or []
        self.vocab = vocab
        self.merges = merges
        self.rev_vocab = {vocab: id for id, vocab in self.vocab.items()}

    def encode(self, text: str) -> list[int] : 
        '''Encode an input text into a sequence of token IDs. 
        For the encoder you don't need to build a work frequency dict. Every computation is done at the word-level.'''
        
        list_integer = []
        segments = []
        
        # similar code segmentation than the training BPE implementation.
        if self.special_tokens:
            special_re = re.compile("(?V1)(" + "|".join(
                re.escape(s) for s in sorted(set(self.special_tokens), key=len, reverse=True)
            ) + ")")
            segments = [s for s in special_re.split(text) if s]   
        else:
            segments = [text]
        
        for seg in segments: 
            if seg in self.special_tokens:
                list_integer.append(self.rev_vocab[seg.encode("utf-8")])
                continue
          
            # STEP 1: Pre-tokenize the sequence
            pretokenize = list(re.finditer(PAT_RE, seg))  

            # STEP 2: represent each token as a UTF-8 bytes. 
            utf8_bytes = [m.group(0).encode("utf-8") for m in pretokenize]
            
            # STEP 3: create a rank dict
            '''the pair with the best (lowest) rank in the merges list). 
            Why the lowest ? because that’s the one that would have been merged first during training, and the encoding reproduces the trainiing process.
            The merging doewsn't happen from left to right but by the pair with the lowest rank first.
            Let's create a rank dict for elements in merge: rank_dict contains the rank of each element in merges'''
            
            rank_dict = {pair: r for r, pair in enumerate(self.merges)}

            #STEP 4 : identify the first applicable merge 
            for encoded in utf8_bytes:
                chunks = [bytes([b]) for b in encoded]
            
                while True:
                    best_pair_tuple = None        
                    best_rank = float("inf")
                    
                # STEP 5: Fetch the next merge: greedy by rank.
                    for consecutive_byte_pair in zip(chunks, chunks[1:]):        
                        rank = rank_dict.get(consecutive_byte_pair)
                        if rank is not None and rank < best_rank:
                            best_pair_tuple = consecutive_byte_pair
                            best_rank = rank
                
                    if best_pair_tuple is None:
                        break 
                        
                    merged = []
                    index = 0
                    changed = False

                    #same principle as BPE training
                    while index < len(chunks):
                        if index + 1 < len(chunks) and chunks[index] == best_pair_tuple[0] and chunks[index + 1] == best_pair_tuple[1]:
                            merged.append(chunks[index] + chunks[index + 1])
                            index += 2
                            changed = True
                        else:
                            merged.append(chunks[index])
                            index += 1
                            
                    if not changed:
                        break  
                    
                    chunks = merged
                
                for element in chunks:  
                    list_integer.append(self.rev_vocab[element])  

        return list_integer
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int] :    
        '''Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.'''     
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id
                 
    def decode(self, ids: list[int]) -> str : 
        '''Decode a sequence of token IDs into text.'''
        list_bytes = []
        
        # STEP 1 : look up each ID’s corresponding entries in the vocabulary (a byte sequence), concatenate them together.
        list_bytes = [self.vocab[i] for i in ids]

        # STEP 2 : decode the bytes to a Unicode string. 
        string = b"".join(list_bytes)
        return string.decode("utf-8", errors="replace")

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f: vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f: merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

if __name__ == "__main__":
    bpe = BPE(special_tokens=["<|endoftext|>"])
    vocab, merges = bpe.train("/Users/akouhana/CS336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt", vocab_size=1000)
    tokenizer = BPE_Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
    tokenizer.encode("Coding BPE from scratch was fun.")
    tokenizer.encode_iterable("Coding BPE is meh.")
    tokenizer.decode([89, 289, 298, 302, 452, 422, 767, 642, 111, 120])