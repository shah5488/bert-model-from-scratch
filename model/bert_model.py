import torch.nn as nn
from model.bert_embeddings import BertEmbeddings
from model.transformer_layer import TransformerLayer


class BertModelFromScratch(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072
    ):
        super().__init__()

        self.embeddings = BertEmbeddings(vocab_size, hidden_size)

        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])

    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, attention_mask)

        return x
