from typing import List, Dict, Optional, Union, Tuple, Any
import regex as re
from collections import defaultdict
import cProfile
from pretokenization_example import find_chunk_boundaries


def read_file(input_path: str, max_chars: int = 220000) -> str:
    """Read first max_chars from file
    """
    with open(input_path, 'r', encoding='utf-8') as file:
        content = file.read(max_chars)  # reads only first max_char characters (prevents overloading memory)
    return content


def train_bpe(input_path: str, vocab_size: int = 256, special_tokens: list[str] = None, max_iteration: int  = 100) : 
    '''
    Trains a (byte-level) BPE tokenizer on a text. 
    Returns: 
        - Vocab: dict[int, bytes] which contains the tokenizer vocabulary.
        - Merges: list[tuple[bytes, bytes]] which contains the list of BPE merges produced from training.
    '''

    text_content = read_file(input_path)
    
    if special_tokens is None: 
        special_tokens = ["<|endoftext|>"] 
    
    # STEP 1: removing special tokens
    text_content = text_content.lower()
    cleaned_text = re.sub(r'[!@#$\^*]', ' ', text_content)
    
    # STEP 2: pre-tokenize 
   
    # a) split text into chunks
    escaped_tokens = [re.escape(token) for token in special_tokens]
    split_pattern = "|".join(escaped_tokens)
    chunks = re.split(split_pattern, cleaned_text)

    # b) utf-8 encoding
    pretokens = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #Pre-tokenization pattern
    for chunk in chunks:
        pretokens.extend(re.finditer(PAT, chunk)) # convert input text into pre-token
    pretokens = [strings.group(0) for strings in pretokens]

    utf8_encoded = [pretoken.encode("utf-8") for pretoken in pretokens] # rpz each token as utf-8 bytes.
    
    # c) count word occurences
    word_frequency_dict = {}
    for word in utf8_encoded: 
        word_as_tuple = tuple(bytes([byte]) for byte in word)
        word_frequency_dict[word_as_tuple] = word_frequency_dict.get(word_as_tuple,0) + 1 
  
    # STEP 3: merge the most likely pairs 

    # a) count pair occurences 
    byte_pair_frequency = defaultdict(int) 
    for word in word_frequency_dict:
        for consecutive_byte_pair in zip(word, word[1:]):
            byte_pair_frequency[consecutive_byte_pair] +=  word_frequency_dict[word]

    # b) merge them 
    merges = []
    
    sorted_dict = sorted(byte_pair_frequency.items(), key=lambda item: item[1], reverse = True)

    for i in range(max_iteration): 
        most_frequent_pair = sorted_dict[-i][0] 
        most_frequent_pair = b''.join(most_frequent_pair) #from tuple to pair 
        merges.append(most_frequent_pair) 
        
        vocab_size += 1  

    return byte_pair_frequency, merges

# cProfile.run('re.compile("foo|bar")')
train_bpe("/Users/akouhana/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")