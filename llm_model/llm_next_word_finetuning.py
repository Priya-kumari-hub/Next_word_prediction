!pip install -q transformers datasets accelerate torch

import torch
print(torch.cuda.is_available())

from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("distilgpt2")
model.resize_token_embeddings(len(tokenizer))

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

tokenized_dataset = tokenized_dataset.map(
    lambda x: {"labels": x["input_ids"]},
    batched=True
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=100,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()

from google.colab import drive
drive.mount('/content/drive')

SAVE_PATH = "/content/drive/MyDrive/llm_next_word_project"
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

tokenizer = GPT2Tokenizer.from_pretrained(SAVE_PATH)
model = GPT2LMHeadModel.from_pretrained(SAVE_PATH)

model.to(device)
model.eval()

def predict_next_words_llm(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(predict_next_words_llm("India is"))
print(predict_next_words_llm("Technology will"))
