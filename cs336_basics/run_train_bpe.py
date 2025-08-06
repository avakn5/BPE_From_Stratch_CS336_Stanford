from typing import List, Dict, Optional, Union, Tuple, Any
import regex as re
from collections import defaultdict
import cProfile

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #Pre-tokenization pattern

def read_file(input_path: str, max_chars: int = 220000) -> str:
   
    """Read first max_chars from file
    """
   
    with open(input_path, 'r', encoding= "utf-8") as file:
        content = file.read(max_chars)  # reads only first max_char characters (prevents overloading memory)
    return content

def merge_pair_in_dict( pair_to_merge: tuple[bytes, bytes], dicti : dict[tuple[bytes], int], special_tokens: list[str]) -> dict[tuple[bytes], int]:
    
    """
    Returns a new dictionary with the specified pair merged in all words.

    Args:
        pair_to_merge: The byte pair to merge (e.g., (b't', b'h')).
        word_freq_dict: A dictionary mapping words (tuples of bytes) to frequencies.

    Returns:
        A new dictionary with the pair merged in each word where applicable.
    """

    new_dict = {}
    
    for word, freq in dicti.items() :  
        
        # keep special tokens untouched
        if b''.join(word) in special_tokens:
            new_dict[word] = freq
            continue
        
        newlist = [] # a tuple can't be modified in place so we need to create a list to store the new tuple containing the merged pair.
        index = 0 
        
        while index < len(word)-1 : 
            string_to_merge = b"".join(pair_to_merge)
            
            if ( (word[index], word[ index+1]) == pair_to_merge): # if the pair to merge is contained in the word
                newlist.append(string_to_merge) #we append to the new list the merged string.
                index += 2 # we skip 2 characters because we just merged two characters

            else: 
                newlist.append(word[index]) # we append the characters that are not part of the pair to merge.
                index += 1 
                
        if index == len(word)-1: #handling the last character separately
            newlist.append(word[index])
            
        new_dict[tuple(newlist)] = freq # we store the frequence of the word back to the new dict
    
    return new_dict
    

def train(input_path: str, vocab_size: int, special_tokens: list[str] = None) : 
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
    special_tokens = [b"<|endoftext|>"]
    
    # STEP 1: removing special characters
    text_content = text_content.lower()
    
    #substitutes special characters with space in the text_content
    cleaned_text = re.sub(r'[!@#$\^*]', ' ', text_content) 
    
    # STEP 2: pre-tokenize : split text into chunks
    
    # the special token |<endoftext>| is "escaped" meaning that it is set aside in the tokenization scheme 
    # otherwise  |<>| would be interpreted as a metacharacter, and not used as a text delimiter. 
    escaped_tokens = [re.escape(token.decode("utf-8")) for token in special_tokens]
   
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
    
    # a) initialize the vocabulary
    vocabulary = {i: bytes([i]) for i in range(256)}

        
    for token in special_tokens: #special tokens need to be added to the vocabulary
        vocabulary[len(vocabulary)] = token

    merges = [] # merge them 3
    pair_frequency_dict = defaultdict(int) # count pair occurences 
    
    while len(vocabulary) < vocab_size:
 
        # Clear previous counts before recalculating for the current state of word_frequency_dict
        pair_frequency_dict.clear() 
        # b) compute byte_pair_frequency from word_frequency_dict.

        for word in word_frequency_dict:
            for consecutive_byte_pair in zip(word, word[1:]): #looking for every pair of adjacent characters.
                # adding the pair frequency to the byte_pair_frequency dictionary. 
                pair_frequency_dict[consecutive_byte_pair] +=  word_frequency_dict[word] #here we get the frequency of each pair across the chunk.
               
        # we need to break this while loop in case of a small text where we would never reach the vocab size.
        if not pair_frequency_dict:
            print("No more pairs to merge. Early Stop.")
            break

        # c) find the most frequent pair
        sorted_dict = sorted(pair_frequency_dict.items(), key=lambda item: item[1], reverse = True) # sort by frequency in descending order
        most_frequent_tuple = sorted_dict[0][0] # fect the most frequent pair

        # d) update word_frequency_dict : merge the most frequent typle with a single token. 
        word_frequency_dict = merge_pair_in_dict(most_frequent_tuple, word_frequency_dict, special_tokens)
        
        # e) append the merged pair to the merges list.
        merges.append(most_frequent_tuple)   
        vocabulary[len(vocabulary)] = b"".join(most_frequent_tuple)
    
    return vocabulary, merges

# cProfile.run('re.compile("foo|bar")')
vocab, merges = train("/Users/akouhana/CS336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt", 1000, special_tokens=["<|endoftext|>"])

# Check that the special token is not in the vocab
vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes