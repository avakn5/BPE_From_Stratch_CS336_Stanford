from typing import List, Dict, Optional, Union, Tuple, Any
import regex as re
from collections import defaultdict
import cProfile
from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #Pre-tokenization pattern

def read_file(input_path: str, max_chars: int = 220000) -> str:
   
    """Read first max_chars from file
    """
   
    with open(input_path, 'r') as file:
        content = file.read(max_chars)  # reads only first max_char characters (prevents overloading memory)
    return content


def train_bpe(input_path: str, vocab_size: int = 256, special_tokens: list[str] = None, max_iteration: int  = 100) : 
    '''
    Trains a (byte-level) BPE tokenizer on any given text. 
    
    Args:
        - input_path: str Path to a text file with BPE tokenizer training data.
        - vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        - special_tokens: list[str] A list of strings to add to the vocabulary. 

    Returns: 
        - vocab: dict[int, bytes] which contains the tokenizer vocabulary.
        - merges: list[tuple[bytes, bytes]] which contains the list of BPE merges produced from training.
    '''
    
    text_content = read_file(input_path) 
    
    if special_tokens is None: 
        special_tokens = ["<|endoftext|>"] 
    
    # STEP 1: removing special characters
    text_content = text_content.lower()
    
    #substitutes special characters with space in the text_content
    cleaned_text = re.sub(r'[!@#$\^*]', ' ', text_content) 
    
    # STEP 2: pre-tokenize : split text into chunks
    
    # the special token |<endoftext>| is "escaped" meaning that it is set aside in the tokenization scheme 
    # otherwise  |<>| would be interpreted as a metacharacter, and not used as a text delimiter. 
    escaped_tokens = [re.escape(token) for token in special_tokens]
   
    # creates the split pattern  
    split_pattern = "|".join(escaped_tokens)
    
    # re.split() breaks the text at each occurrence of the pattern
    chunks = re.split(split_pattern, cleaned_text) 
    
    # b) utf-8 encoding
    pretokens = []
    for chunk in chunks:
        #chunks are pre-tokenized (using PAT, the pre-tokenization pattern), and added to the end of the list pretokens.
        pretokens.extend(re.finditer(PAT, chunk)) 

    #extracting only the pretokens from the regex object
    pretokens = [strings.group(0) for strings in pretokens] 

    #encodes each pretoken in utf-8 encoding.
    utf8_encoded = [pretoken.encode("utf-8") for pretoken in pretokens] 
    
    # c) count word occurences
    word_frequency_dict = {}
    for word in utf8_encoded: 
        # creating a dictionary containing words stored as tuples and their frequency in the text.
        word_as_tuple = tuple(bytes([byte]) for byte in word)
        # fetching the frequency of the word in the text. 0 being the default value, if the word is not found in the text. 
        word_frequency_dict[word_as_tuple] = word_frequency_dict.get(word_as_tuple, 0) + 1 

    # STEP 3: merge the most likely pairs 

    # a) count pair occurences 
    byte_pair_frequency = defaultdict(int) 
    
    for word in word_frequency_dict:
        for consecutive_byte_pair in zip(word, word[1:]): #looking for every pair of adjacent characters.
            # adding the pair frequency to the byte_pair_frequency dictionary. 
            byte_pair_frequency[consecutive_byte_pair] +=  word_frequency_dict[word]

    # b) merge them 
    merges = []
    sorted_dict = sorted(byte_pair_frequency.items(), key=lambda item: item[1], reverse = True) # sort by frequency in increasing order

    #while current_vocab_size < vocab_size
    for i in range(max_iteration): 
        most_frequent_pair = sorted_dict[-i][0] # fect the most frequent pair
        most_frequent_pair = b''.join(most_frequent_pair) # merge the most frequent pair to be a string instead of a tuple
        merges.append(most_frequent_pair) 
        vocab_size += 1  #update the vocabulary size.
         
    # c) constructing the tokenizer vocabulary 
    vocabulary = {} 
    for id in enumerate(merges):
        vocabulary[id] = id
    
    assert len(vocabulary) == len(set(vocabulary)) # vocabulary should be a set

    return vocabulary, merges

# cProfile.run('re.compile("foo|bar")')
train_bpe("/Users/akouhana/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt")