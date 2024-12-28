from datasets import load_dataset

# datasets_list = list_datasets()
# print(f"Available datasets: {len(datasets_list)}")

dataset = load_dataset("wikitext", "wikitext-103-v1")
print(dataset)
