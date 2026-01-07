from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

SAVE_PATH = "/content/drive/MyDrive/llm_next_word_project"

tokenizer = GPT2Tokenizer.from_pretrained(SAVE_PATH)
model = GPT2LMHeadModel.from_pretrained(SAVE_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
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
            temperature=0.8
        )

    return tokenize
  r.decode(output[0], skip_special_tokens=True)

print(predict_next_words_llm("India is"))
// Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
India is a large island of temperate waters , with a range of temperate and temperate systems in which the sea ranges from a low point of the Earth 's crust to a maximum depth of 30 m ( 27 ft ) . The climate is temperate

print(predict_next_words_llm("Technology will"))
// Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Technology will allow AI to increase in areas where AI is not sufficiently capable to compete in the marketplace . AI is not alone in this arena , but it will continue to improve in other fields , and in other areas where AI is not sufficiently capable to compete in the
