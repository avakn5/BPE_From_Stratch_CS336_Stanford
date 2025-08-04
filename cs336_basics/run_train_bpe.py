from typing import List, Dict, Optional, Union, Tuple, Any
import regex as re
from collections import defaultdict


def read_file(input_path: str, max_chars: int = 10000) -> str:
    """Read first max_chars from file"""
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read(max_chars)  # reads only first 10k characters
    return content

def train_bpe(input_path: str, vocab_size: int = 256, special_tokens: list[str] = None) : 
 
    '''
    Trains a (byte-level) BPE tokenizer on a text. 
    The function returns 
        vocab: dict[int, bytes] and merges: list[tuple[bytes, bytes]]
    '''

    text = read_file(input_path)
    
    if special_tokens is None: 
        special_tokens = ["<|endoftext|>"] 
    
    # STEP 1: removing special tokens
    text = text.lower()
    text = re.sub(r'[!@#$\^*]', ' ', text)
    
    # STEP 2: pre-tokenize 
   
    # a) split on special tokens
    tokens = re.findall(r"<\|endoftext\|>|[^\s]+", text) # convert input text into pre-token
    utf8_encoded = [token.encode("utf-8") for token in tokens] # rpz each token as utf-8 bytes.
    
    # b) count occurences
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
    sorted_dict = sorted(pair_dict) #to change

    for i in range(10):
        pairmerge = sorted_dict[-i]
        pairmerge = b''.join(pairmerge) #from tuple to pair 
        merges.append(pairmerge) 
        vocab_size += 1  
    
    
    print(vocab_size)
    print(merges)
    return pair_dict, merges

train_bpe("/Users/akouhana/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")