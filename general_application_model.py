from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
# Function to load and Tokenize dataset
def prepare_dataset():
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Loading tokenizer and set padding token
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding=True), batched=True
    )
    return tokenized_dataset, tokenizer
# Prepare the model
def load_model_and_config(tokenizer):
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    return model
#Trianing Arguments
def prepare_training_args():
    training_args = TrainingArguments(
        output_dir="./results",             #Path to save model and logs
        overwrite_output_dir=True,          #Overwrite previous runs
        num_train_epochs=3,                 #Number of training epochs
        per_device_train_batch_size=4,      #Training batch size per GPU/CPU
        per_device_eval_batch_size=8,       #Evaluation batch size
        warmup_steps=500,                   #Warming stes for learning rate scheduler
        weight_decay=0.01,                  #Regularization to avoid overfitting
        logging_dir="./logs",               #Directory for logging
        save_steps=10_000,                  #Save checkpoint every 10k step
        save_total_limit=2,                 #Keep only 2 most recent checkpoints
        
    )
    return training_args

def train_model():
    tokenized_dataset, tokenizer = prepare_dataset()
    #Train and validate dataset
    train_dataset = tokenized_dataset["train"]
    val_dataset = tokenized_dataset["validation"]
    # Loadng the model
    model = load_model_and_config(tokenizer)
    #Prepare the datacollector
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    #Preparing Training Arguments
    training_args = prepare_training_args()
    #Trainer Initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset = val_dataset,
        data_collator=data_collator,
    )
    #Train the model
    trainer.train()
    #saving the model and tokenizer
    trainer.save_model("./trained_model")
    tokenizer.save_pretrained("./trained_model")
    print("Training compete and model saved")

if __name__ == "__main__":
    train_model()    