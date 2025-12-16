```text
bert-from-scratch/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── processed
│   ├── saw_texts
│   ├── load_text_dataset.py
│
├── tokenizer/
│   ├── __init__.py
│   ├── inspect_token_func.py
│   ├── inspect_tokenization.py
│   ├── wordpiece_tokenizer.py
│
├── model/
│   ├── __init__.py
│   ├── bert_embeddings.py
│   ├── bert_model.py
│   ├── compare_hf_vs_scratch.py
│   ├── inspect_attention_maps.py
│   ├── inspect_bert_embeddings_layer.py
│   ├── inspect_scratch_attention.py
│   ├── self_attention.py
│   ├── transformer_layer.py
│
├── pretraining/
│   ├── __init__.py
│   ├── mlm_dataset.py
│   ├── mlm_masking.py
│   ├── mlm_pretraining_step.py
│   ├── pretrain_bert.py
│
├── training/
│   ├── bert_scratch_classifier.py
│   ├── classification_dataset.py
│   ├── simple_bert_classifier.py
│   ├── train_classifier.py
│
├── inference/
│   ├── predict.py
│   ├── inspect_hidden_states.py
│
└── notebooks/
    ├── 001_token_flow.ipynb
    ├── 002_attention_visualization.ipynb
    └── 003_cls_evolution.ipynb
```


This repository explains and implements **BERT from scratch** using PyTorch.
The goal is **concept clarity**, not performance. Conceptual clarity gets lost in the midst of fancy youtube videos.

## What concepts you will learn
- How tokenization works
- How embeddings are built
- How self-attention works step-by-step
- How MLM & NSP pretraining works
- How training loop is structured

## What this repo is NOT
- Not HuggingFace
- Not optimized for production
- Not meant to replace pretrained BERT

## Architecture Overview
```text
Simple diagram

                ┌───────────────────────┐
                │       Raw Text        │
                │  (sentences, docs)   │
                └───────────┬──────────┘        
                            │
                            V
        ┌─────────────────────────────────────────┐
        │              Tokenizer                  │
        │                                         │
        │  Basic Tokenizer  →  WordPiece          │
        │  (split text)        (subwords)         │
        │                                         │
        │  Adds: [CLS] [SEP] [PAD]                │
        └──────────────────┬──────────────────────┘
                           │
                           V
        ┌─────────────────────────────────────────┐
        │        Model Input Representation       │
        │                                         │
        │  input_ids                              │
        │  attention_mask                         │
        │  token_type_ids                         │
        └─────────────────┬───────────────────────┘
                          │
                          V
        ┌─────────────────────────────────────────┐
        │             Embedding Layer             │
        │                                         │
        │  Token Embeddings                       │
        │  Position Embeddings                    │
        │  Segment Embeddings                     │
        │                                         │
        │  (Sum + LayerNorm + Dropout)            │
        └───────────┬─────────────────────────────┘
                    │
                    V
        ┌─────────────────────────────────────────┐
        │        Transformer Encoder Stack        │
        │                                         │
        │  ┌───────────────────────────────────┐  │
        │  │  Self-Attention (Multi-Head)      │  │
        │  │  Feed Forward Network             │  │
        │  │  Residual + LayerNorm             │  │
        │  └───────────────────────────────────┘  │
        │                 × N layers              │
        └───────────┬─────────────────────────────┘
                    │
                    V
        ┌─────────────────────────────────────────┐
        │        Contextual Representations       │
        │                                         │
        │  Token-level embeddings                 │
        │  CLS embedding (sequence summary)       │
        └───────────────┬─────────────────────────│
                        │
        ┌───────────────┴──────────────────┐
        V                                  V
┌─────────────────────────┐   ┌────────────────────────┐
│  Pretraining            │   │  Downstream Training   │
│                         │   │  (Classification)      │
│  MLM Head               │   │                        │
│  NSP Head               │   │  CLS → Linear → Softmax│
└─────────────────────────┘   └────────────────────────┘
                    │
                    V
        ┌─────────────────────────────────────────┐
        │          Inference & Analysis           │
        │                                         │
        │  Predictions                            │
        │  Attention inspection                   │
        │  CLS evolution analysis                 │
        └─────────────────────────────────────────┘
```


## How to run
```bash
pip install -r requirements.txt
python training/train.py
```

