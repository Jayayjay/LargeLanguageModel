from datasets import load_dataset
from transformers import AutoTokenizer

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Load the GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set eos_token as the pad_token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset with padding and truncation
tokenized_dataset = dataset.map(
    lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True
)

# Save the tokenized dataset
tokenized_dataset.save_to_disk("./tokenized_wikitext")
