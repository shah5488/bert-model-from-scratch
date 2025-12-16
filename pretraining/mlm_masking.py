import torch
import random


def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    """
    Implements BERT-style MLM masking (80/10/10)
    """
    labels = input_ids.clone()

    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
    )

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # loss ignored

    # 80% [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    )
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% random token
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(
        tokenizer.vocab_size, labels.shape, dtype=torch.long
    )
    input_ids[indices_random] = random_words[indices_random]

    # 10% unchanged → nothing to do

    return input_ids, labels
