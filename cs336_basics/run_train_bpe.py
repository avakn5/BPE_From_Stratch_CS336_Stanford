from typing import List, Dict, Optional, Union, Tuple, Any
import regex as re
from collections import defaultdict
import cProfile


def read_file(input_path: str, max_chars: int = 220000) -> str:
    """Read first max_chars from file
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read(max_chars)  # reads only first 10k characters
    return content



def train_bpe(input_path: str, vocab_size: int = 256, special_tokens: list[str] = None, iteration: int  = 100) : 
    '''
    Trains a (byte-level) BPE tokenizer on a text. 
    Returns: 
        - vocab: dict[int, bytes] which contains the tokenizer vocabulary.
        - merges: list[tuple[bytes, bytes]] which contains the list of BPE merges produced from training.
    '''

    text = read_file(input_path)
    
    if special_tokens is None: 
        special_tokens = ["<|endoftext|>"] 
    
    # STEP 1: removing special tokens
    text = text.lower()
    text = re.sub(r'[!@#$\^*]', ' ', text)
    
    # STEP 2: pre-tokenize 
   
    # a) split text into chunks
    escaped_tokens = [re.escape(token) for token in special_tokens]
    split_pattern = "|".join(escaped_tokens)
    chunks = re.split(split_pattern, text)

    # b) utf-8 encoding
    tokens = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        tokens.extend(re.finditer(PAT, chunk)) # convert input text into pre-token
    tokens = [strings.group(0) for strings in tokens]

    utf8_encoded = [token.encode("utf-8") for token in tokens] # rpz each token as utf-8 bytes.
    
    # b) count word occurences
    char_dict = {}
    for word in utf8_encoded: 
        word_as_tuple = tuple(bytes([byte]) for byte in word)
        char_dict[word_as_tuple] = 1 if char_dict.get(word_as_tuple) is None else char_dict[word_as_tuple] + 1    

    # STEP 3: merge the most likely pairs 

    # a) count pair occurences 
    pair_dict = defaultdict(int) 
    for word in char_dict:
        for pair in zip(word, word[1:]):
            pair_dict[pair] +=  char_dict[word]
    
    # b) merge them 
    merges = []
    sorted_dict = sorted(pair_dict.items(), key=lambda item: item[1], reverse = True)

    for i in range(iteration):
        pairmerge = sorted_dict[-i][0]
        pairmerge = b''.join(pairmerge) #from tuple to pair 
        merges.append(pairmerge) 
        vocab_size += 1  
        
    print(merges)
        
    return pair_dict, merges

cProfile.run('re.compile("foo|bar")')
train_bpe("/Users/akouhana/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")