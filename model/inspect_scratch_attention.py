import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer
from model.bert_model import BertModelFromScratch
from model.self_attention import SelfAttention

# Monkey-patch to capture attention scores
original_forward = SelfAttention.forward

def forward_with_attention(self, x, attention_mask=None):
    batch_size, seq_len, _ = x.size()

    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)

    Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
    attention_probs = torch.softmax(scores, dim=-1)

    self.last_attention = attention_probs.detach()

    context = torch.matmul(attention_probs, V)
    context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    return self.out(context)

SelfAttention.forward = forward_with_attention

# ------------------------------
# Run model
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "Artificial intelligence improves decision making in businesses"

inputs = tokenizer(text, return_tensors="pt", max_length=16, truncation=True)
model = BertModelFromScratch(num_layers=1)

_ = model(inputs["input_ids"], inputs["attention_mask"])

attn = model.layers[0].attention.last_attention[0, 0]

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

plt.figure(figsize=(8, 6))
plt.imshow(attn)
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.title("Scratch BERT Attention (Layer 1, Head 1)")
plt.colorbar()
plt.show()
