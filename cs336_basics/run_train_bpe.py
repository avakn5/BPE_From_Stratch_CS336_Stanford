from typing import List, Dict, Optional, Union, Tuple, Any
import regex as re
from collections import defaultdict
import cProfile
from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #Pre-tokenization pattern

def read_file(input_path: str, max_chars: int = 220000) -> str:
   
    """Read first max_chars from file
    """
   
    with open(input_path, 'r', encoding= "utf-8") as file:
        content = file.read(max_chars)  # reads only first max_char characters (prevents overloading memory)
    return content

def merge_pair_in_dict( pair_to_merge: tuple[bytes, bytes], dictionary: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    
    """
    Returns a new dictionary with the specified pair merged in all words.

    Args:
        pair_to_merge: The byte pair to merge (e.g., (b't', b'h')).
        word_freq_dict: A dictionary mapping words (tuples of bytes) to frequencies.

    Returns:
        A new dictionary with the pair merged in each word where applicable.
    """
    return dictionary
    
    

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str] = None) : 
    '''
    Trains a (byte-level) BPE tokenizer on any given text. 
    
    Args:
        - input_path: str Path to a text file with BPE tokenizer training data.
        - vocab_size: int A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        - special_tokens: list[str] A list of strings to add to the vocabulary. 

    Creates: 
        - word_frequency_dict : contains the frequency of each word and is useful to count pair frequencies (which are afterwards stored in the byte_pair_frequency dict). The merged pairs are modified in place inside the word_frequnecy_dict.
        - pair_frequency_dict : temprary helper which contains the frequency of each pair. It is recomputed at each iteration. 
    Returns: 
        - vocab: dict[int, bytes] which contains the tokenizer vocabulary.
        - merges: list[tuple[bytes, bytes]] which contains the list of BPE merges produced from training.
    '''
    
    text_content = read_file(input_path) 
    
    if special_tokens is None: 
        special_tokens = ["<|endoftext|>"] 
    
    current_vocab_size = 256

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
    
    # c) creating the word frequency dict
    
    word_frequency_dict = {}    #  count word occurences
    for word in utf8_encoded: 
            # creating a dictionary containing words stored as tuples and their frequency in the text.
            word_as_tuple = tuple(bytes([byte]) for byte in word)
            # fetching the frequency of the word in the text. 0 being the default value, if the word is not found in the text. 
            word_frequency_dict[word_as_tuple] = word_frequency_dict.get(word_as_tuple, 0) + 1 
            
    # STEP 3: merge the most likely pairs 
    
    pair_frequency_dict = defaultdict(int) # count pair occurences 
    merges = [] # merge them 3
    vocabulary = {} 
    
    while current_vocab_size < vocab_size:
        
        # a) compute byte_pair_frequency from word_frequency_dict.

        for word in word_frequency_dict:
            for consecutive_byte_pair in zip(word, word[1:]): #looking for every pair of adjacent characters.
                # adding the pair frequency to the byte_pair_frequency dictionary. 
                pair_frequency_dict[consecutive_byte_pair] +=  word_frequency_dict[word] #here we get the frequency of each pair across the chunk.
        
        # b) find the most frequent pair
        sorted_dict = sorted(pair_frequency_dict.items(), key=lambda item: item[1], reverse = True) # sort by frequency in descending order
        most_frequent_tuple = sorted_dict[0][0] # fect the most frequent pair
        most_frequent_string = b''.join(most_frequent_tuple) # merge the most frequent pair to be a string instead of a tuple

        # c) update word_frequency_dict : merge the most frequent typle with a single token. 
        word_frequency_dict = merge_pair_in_dict(most_frequent_tuple, word_frequency_dict)
        
        # d) append the merged pair to the merges list.
        merges.append(most_frequent_string)
        
    return vocabulary, merges

# cProfile.run('re.compile("foo|bar")')
train_bpe("/Users/akouhana/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 500)