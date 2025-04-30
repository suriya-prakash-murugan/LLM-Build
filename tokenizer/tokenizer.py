import re
import collections
import json
from pathlib import Path

class ValyrianBPETokenizer:
    def __init__(self):
        self.vocab_size = 500
        self.vocab = {}
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}


    def get_vocab(self, corpus):
        # Split text into words and represent each word as a list of characters + </w>
        tokens = corpus.split()
        vocab = collections.defaultdict(int)
        for word in tokens:
            word_tuple = tuple(word) + ('</w>',)
            vocab[word_tuple] += 1
        return vocab
    
    def get_stats(self, vocab):
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        pattern = re.escape(' '.join(pair))
        pattern = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')
        for word, freq in vocab.items():
            word_str = ' '.join(word)
            word_str = pattern.sub(''.join(pair), word_str)
            new_word = tuple(word_str.split())
            new_vocab[new_word] = freq
        return new_vocab
    
    def train(self, corpus):
        vocab = self.get_vocab(corpus)
        for i in range(self.vocab_size):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.merges.append(best)
        self.vocab = vocab
        # Build final vocab after all merges
        all_tokens = set()
        for word_tuple in self.vocab:
            all_tokens.update(word_tuple)

        self.token_to_id = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}


    def encode_word(self, word):
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            mergeable = [p for p in pairs if p in self.merges]
            if not mergeable:
                break
            bigram = min(mergeable, key=lambda p: self.merges.index(p))
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word
    
    def encode(self, text):
        encoded_words = [self.encode_word(word) for word in text.strip().split()]
        # Flatten and map to token IDs
        token_ids = [self.token_to_id[token] for word in encoded_words for token in word]
        return token_ids

    def decode(self, token_ids):
        tokens = [self.id_to_token[i] for i in token_ids]
        # Reconstruct words from merged tokens
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

    
    def save(self, path):
        path = Path(path)
        with open(path / 'merges.json', 'w') as f:
            json.dump(self.merges, f)
        with open(path / 'vocab.json', 'w') as f:
            json.dump(self.token_to_id, f)


    def load(self, path):
        path = Path(path)
        with open(path / 'merges.json', 'r') as f:
            self.merges = [tuple(pair) for pair in json.load(f)]
        with open(path / 'vocab.json', 'r') as f:
            self.token_to_id = json.load(f)
            self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
