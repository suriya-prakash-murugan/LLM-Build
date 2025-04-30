import sys
sys.path.append("./")
from tokenizer import ValyrianBPETokenizer
from pathlib import Path

corpus_path = Path("data/High Valyrian.txt")
output_path = Path("tokenizer/")

# Load corpus
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus = f.read()

# Train BPE Tokenizer
tokenizer = ValyrianBPETokenizer()
tokenizer.train(corpus)

# Save merges (learned rules)
tokenizer.save(output_path)

# Test encoding and decoding
print("\n--- Sample Encoding ---")
test_sentence = "Valar Morghulis"
encoded = tokenizer.encode(test_sentence)
print("Encoded:", encoded)
print("Decoded:", tokenizer.decode(encoded))