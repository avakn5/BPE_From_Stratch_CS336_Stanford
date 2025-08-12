import regex as re
from collections import defaultdict
import cProfile, pstats
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" #Pre-tokenization pattern
PAT_RE = re.compile(PAT)

class BPE():
    
    def __init__(self, special_tokens: list[str]):
        self.special_tokens = special_tokens or []
        self.vocabulary: dict[int, bytes] = {}
        self.merges: list[tuple[bytes, bytes]] = []
   
        
    # def read_file_max_char(self, input_path: str,max_chars: int) -> str:
    #     """Read file content."""
    #     with open(input_path, 'r', encoding= "utf-8") as file:
    #         return file.read(max_chars)   
    
    def read_file(self, input_path: str) -> str:
        """Read file content."""
        with open(input_path, 'r', encoding= "utf-8") as file:
            return file.read()   

    def merge_pair_in_dict(self, pair_to_merge: tuple[bytes, bytes], word_freq_dict : dict[tuple[bytes], int], special_tokens_b: list[bytes]) -> dict[tuple[bytes], int]:
        
        """
        Returns a new dictionary with the specified pair merged in all words.

        Args:
            pair_to_merge: The byte pair to merge (e.g., (b't', b'h')).
            word_freq_dict: A dictionary mapping words (tuples of bytes) to frequencies.

        Returns:
            A new dictionary with the pair merged in each word where applicable.
            """
        first_byte, second_byte = pair_to_merge
        merged_bytes = first_byte + second_byte
        new_dict = {}
        special_tokens_b = {tuple(bytes([b]) for b in token) for token in special_tokens_b}

        for word, freq in word_freq_dict.items() :  
            
            # keep special tokens untouched
            if word in special_tokens_b:
                new_dict[word] = freq
                continue
            
            newlist = [] # a tuple can't be modified in place so we need to create a list to store the new tuple containing the merged pair.
            index = 0 
            n = len(word)
            
            while index < n-1 : 
                
                if ( (word[index], word[ index+1] ) == pair_to_merge): # if the pair to merge is contained in the word
                    newlist.append(merged_bytes) #we append to the new list the merged string.
                    index += 2 # we skip 2 characters because we just merged two characters

                else: 
                    newlist.append(word[index]) # we append the characters that are not part of the pair to merge.
                    index += 1 
                    
            if index == n-1: #handling the last character separately
                newlist.append(word[index])
                
            new_dict[tuple(newlist)] = freq # we store the frequence of the word back to the new dict
        
        return new_dict
        

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str] = None) : 
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
        
        text_content = self.read_file(input_path) 
        
        # STEP 1: special tokens encoding
        special_tokens_b = [t.encode("utf-8") if isinstance(t, str) else t for t in self.special_tokens] # encode the special tokens
            
        # STEP 2: pre-tokenize : split text into chunks
        
        # the special token |<endoftext>| is "escaped" meaning that it is set aside in the tokenization scheme         
        escaped_tokens = [re.escape(tok.decode("utf-8")) for tok in special_tokens_b]   # otherwise  |<>| would be interpreted as a metacharacter, and not used as a text delimiter. 

        if escaped_tokens:
            # creates the split pattern  
            split_pattern = "|".join(escaped_tokens)
            # re.split() breaks the text at each occurrence of the pattern
            chunks = re.split(split_pattern, text_content)
        else:
            chunks = [text_content]
            
        # b) utf-8 encoding
        pretokens = []
        for chunk in chunks:
            #chunks are pre-tokenized (using PAT, the pre-tokenization pattern), and added to the end of the list pretokens.
            pretokens.extend(re.finditer(PAT_RE, chunk)) 

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
        self.vocabulary = {i: bytes([i]) for i in range(256)}

        for tok in special_tokens_b:
            if tok not in self.vocabulary.values():
                    self.vocabulary[len(self.vocabulary)] = tok        
            
        self.merges = [] 
        pair_frequency_dict = defaultdict(int) # count pair occurences 
        
        while len(self.vocabulary) < vocab_size:
    
            # Clear previous counts before recalculating for the current state of word_frequency_dict
            pair_frequency_dict.clear() 
            
            # b) compute byte_pair_frequency from word_frequency_dict.      
            for word, freq in word_frequency_dict.items(): #looking for every pair of adjacent characters.
                for consecutive_byte_pair in zip(word, word[1:]):
                    # adding the pair frequency to the byte_pair_frequency dictionary. 
                    pair_frequency_dict[consecutive_byte_pair] += freq  #here we get the frequency of each pair across the chunk.
                
            # we need to break this while loop in case of a small text where we would never reach the vocab size.
            if not pair_frequency_dict:
                print("No more pairs to merge. Early Stop.")
                break

            # c) find the most frequent pair
            most_frequent_tuple = max(pair_frequency_dict.items(), key=lambda item: (item[1], item[0]))[0] #highest count, then highest pair

            # d) update word_frequency_dict : merge the most frequent typle with a single token. 
            word_frequency_dict = self.merge_pair_in_dict(most_frequent_tuple, word_frequency_dict, special_tokens_b)
            
            # e) append the merged pair to the merges list.
            self.merges.append(most_frequent_tuple)   
            self.vocabulary[len(self.vocabulary)] = b"".join(most_frequent_tuple)
                    
        return self.vocabulary, self.merges

class BPE_Tokenizer(): 
    '''Construct a Tokenizer from a given vocabulary, list of merges, and a list of special tokens 
    and uses them to encode and decode text to/from token IDs.'''
     
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None): 
        self.special_tokens = special_tokens or []
        self.vocab = vocab
        self.merges = merges

    def encode(self, text: str) -> list[int] : 
        '''Encode an input text into a sequence of token IDs. 
        For the encoder you don't need to build a work frequency dict. Every computation is done at the word-level.'''
        list_integer = []
        partial_list = []

        # STEP 1: Pre-tokenize the sequence
        pretokenize = list(re.finditer(PAT_RE, text))  

        # STEP 2: represent each token as a UTF-8 bytes. 
        utf8_bytes = [m.group(0).encode("utf-8") for m in pretokenize]
        
        # populate the partial list
        for word in utf8_bytes: 
            for char in word:
                partial_list.append(bytes([char]))

        # STEP 3: create a rank dict
        '''the pair with the best (lowest) rank in the merges list). 
        Why the lowest ? because that’s the one that would have been merged first during training, and the encoding reproduces the trainiing process.
        The merging doewsn't happen from left to right but by the pair with the lowest rank first.
        Let's create a rank dict for elements in merge: rank_dict contains the rank of each element in merges'''
        
        rank_dict = defaultdict(int)
        for rank, pair in enumerate(self.merges):
            rank_dict[pair] = rank
        
        #STEP 4 : identify the first applicable merge 
        for encoded in utf8_bytes:
            
            chunks = [bytes([b]) for b in encoded]
            new_list = []
            best_pair_tuple = None        
            best_rank = float("inf")
            
            while (new_list != partial_list):
            # STEP 5: Fetch the next merge: greedy by rank.
                encoded_as_tuple = tuple(bytes([byte]) for byte in encoded)
                for consecutive_byte_pair in zip(encoded_as_tuple, encoded_as_tuple[1:]):        
                    rank = rank_dict.get(consecutive_byte_pair)
                    if rank is not None and rank < best_rank:
                        best_pair_tuple = consecutive_byte_pair
                        best_rank = rank
            
                    if best_pair_tuple is None:
                        continue 
                    
                    while new_list != partial_list: 
                        merged = []
                        i = 0
                        while i < len(chunks):
                            if i + 1 < len(chunks) and chunks[i] == best_pair_tuple[0] and chunks[i + 1] == best_pair_tuple[1]:
                                merged.append(chunks[i] + chunks[i + 1])
                                i += 2
                            else:
                                merged.append(chunks[i])
                                i += 1
                        chunks = merged
                        
                        # final merged chunks for this word
                        new_list.extend(chunks)
            
            # then it means that no changes 
            print(new_list)
                
        return list_integer
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int] :    
        '''Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.'''     
        pass 
    
    def decode(self, ids: list[int]) -> str : 
        '''Decode a sequence of token IDs into text. '''
        pass
    
    
    def from_files(cls, vocab_filepath : str, merges_filepath: str, special_tokenss: list[str] | None = None): 
        '''Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges.'''
        pass
    
def train(input_path: str, vocab_size: int, special_tokens: list[str] | None = None):
    bpe = BPE(special_tokens=special_tokens)
    return bpe.train(input_path=input_path, vocab_size=vocab_size)

# if __name__ == "__main__":
#     # pr = cProfile.Profile() # profile code's efficiency
#     # pr.runcall(train, "/Users/akouhana/CS336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt", 32000, ["<|endoftext|>"])
#     # pstats.Stats(pr).sort_stats("cumtime").print_stats(30) 
    
    
if __name__ == "__main__":
    bpe = BPE(special_tokens=["<|endoftext|>"])
    vocab, merges = bpe.train("/Users/akouhana/CS336/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt", vocab_size=1000)
    tokenizer = BPE_Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
    tokenizer.encode("Yonatan did too much botox")
