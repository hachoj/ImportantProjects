import regex as re

class BasicTokenizer:

    def __init__(self):
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        # self.vocab = {idx: [idx] for idx in range(256)}

    def train(self, text:str, vocab_size:int, verbose=False):
        num_merge = vocab_size - 255
        tokens = list(text.encode('utf-8'))
        for i in range(num_merge):
            stats = self.get_stats(tokens)
            common_pair = max(stats, key=stats.get) # type: ignore
            tokens = self.combine_tokens(tokens, common_pair, i+256)
            self.merges[common_pair] = i+256
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        new_tokens = list(text.encode('utf-8'))
        while len(new_tokens) >= 2:
            stats = self.get_stats(new_tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            new_tokens = self.combine_tokens(new_tokens, pair, idx)
        return new_tokens 

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text 

    # helper functions
    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def combine_tokens(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i != len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def get_vocab(self):
        for key, value in self.vocab.items():
            print(f'{value.decode('utf-8', errors='replace')}: {key}')

class RegexTokenizer(BasicTokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        re_pat =  re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        num_merge = vocab_size - 255
        text_split = re.findall(re_pat, text)
        token_split = []
        for text in text_split:
            mini_tokens = list(text.encode('utf-8'))
            token_split.append(mini_tokens)
        for i in range(num_merge):
            # allows for stats to be taken with
            # the regex in mind 
            stats = {}
            for token in token_split:
                mini_stats = self.get_stats(token)
                stats = self.merge_dicts_add_values(stats, mini_stats) 
            common_pair = max(stats, key=stats.get) # type: ignore
            for idx, token in enumerate(token_split):
                token_split[idx] = self.combine_tokens(token, common_pair, i+256)
            self.merges[common_pair] = i+256
            for (p0, p1), idx in self.merges.items():
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
            

    # new helper functions
    def merge_dicts_add_values(self, dict1, dict2):
        merged_dict = dict1.copy()  # Start with dict1's keys and values
        for key, value in dict2.items():
            if key in merged_dict:
                merged_dict[key] += value  # Add values together if key exists
            else:
                merged_dict[key] = value
        return merged_dict
    
def main():
    pass

if __name__ == '__main__':
    main()