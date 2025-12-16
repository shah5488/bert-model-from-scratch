from transformers import BertTokenizer

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ----------- Two large text blocks (≈100+ words each) -----------

text_block_1 = """
Artificial intelligence is transforming modern businesses by enabling automation,
better decision-making, and improved customer experiences. Companies across finance,
healthcare, retail, and manufacturing are investing heavily in AI-powered systems.
Machine learning models analyze vast amounts of structured and unstructured data to
identify patterns that humans may overlook. However, challenges such as data quality,
model interpretability, and ethical considerations remain critical concerns.
"""

text_block_2 = """
Sports play a crucial role in promoting physical fitness, mental well-being, and social
interaction. Team sports such as football and cricket teach collaboration and discipline,
while individual sports like tennis and athletics build focus and resilience. Major
international sporting events unite people across cultures, fostering a sense of global
community and shared enthusiasm.
"""

texts = [text_block_1, text_block_2]

# ----------- Tokenization -----------

encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt",
    return_attention_mask=True
)

# ----------- Inspect outputs -----------

print("\n=== TOKEN IDS ===")
print(encoded["input_ids"])

print("\n=== ATTENTION MASK ===")
print(encoded["attention_mask"])

print("\n=== TOKENS (first sample) ===")
tokens_0 = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
print(tokens_0)

print("\n=== SPECIAL TOKENS ===")
print("CLS token:", tokenizer.cls_token)
print("SEP token:", tokenizer.sep_token)
print("PAD token:", tokenizer.pad_token)