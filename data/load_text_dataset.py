from datasets import load_dataset


def load_wiki_sample(split="train"):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    dataset = dataset.filter(lambda x: len(x["text"]) > 100)
    return dataset


if __name__ == "__main__":
    ds = load_wiki_sample()
    print(ds[0]["text"][:300])
