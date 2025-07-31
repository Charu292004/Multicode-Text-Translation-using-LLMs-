from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from aksharamukha import transliterate
import torch

# Load zero-shot classifier for language detection
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

model_name = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto")

def detect_language_zero_shot(text):
    candidate_labels = ["Hindi", "Tamil"]
    result = classifier(text, candidate_labels)
    # returns labels sorted by scores; pick top label
    return result['labels'][0].lower()

def hinglish_to_hindi(text):
    input_text = f"eng_Latn hin_Deva {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True,
            use_cache=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def tanglish_to_tamil(text):
    input_text = f"eng_Latn tam_Taml {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            use_cache=False
        )
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if any('\u0900' <= ch <= '\u097F' for ch in raw_output):
        tamil_output = transliterate.process('Devanagari', 'Tamil', raw_output)
        return tamil_output
    else:
        return raw_output

if __name__ == "__main__":
    print("Auto-detecting Hinglish or Tanglish input using zero-shot classifier...")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("Enter sentence: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        lang = detect_language_zero_shot(user_input)
        if lang == "hindi":
            output = hinglish_to_hindi(user_input)
            print("Detected Hinglish → Hindi Output:", output)
        else:
            output = tanglish_to_tamil(user_input)
            print("Detected Tanglish → Tamil Output:", output)