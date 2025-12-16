```text
bert-from-scratch/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── sample_text.txt
│   └── README.md
│
├── tokenizer/
│   ├── __init__.py
│   ├── basic_tokenizer.py
│   ├── wordpiece.py
│   └── README.md
│
├── model/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── transformer_block.py
│   ├── bert.py
│   └── README.md
│
├── pretraining/
│   ├── __init__.py
│   ├── mlm.py
│   ├── nsp.py
│   ├── dataset.py
│   └── README.md
│
├── training/
│   ├── train.py
│   ├── trainer.py
│   └── README.md
│
├── inference/
│   ├── predict.py
│   └── README.md
│
└── notebooks/
    ├── 01_tokenizer_demo.ipynb
    ├── 02_attention_visualization.ipynb
    └── 03_training_demo.ipynb
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
(Simple diagram or explanation)

## How to run
pip install -r requirements.txt
python training/train.py

