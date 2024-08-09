from fastapi import FastAPI, Form
import re
import torch
from transformers import BertTokenizer, BertForMaskedLM
import logging

logging.basicConfig(level=logging.INFO)  # OPTIONAL

app = FastAPI()

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
model.eval()
# model.to('cuda') #use gpu

@app.post("/predict")
def predict(text: str = Form(...), top_k: int = 5):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')  #use gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    result = []

    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        result.append({"masked_token": predicted_token, "score": float(token_weight), "full_result": re.sub(r"\[MASK\]", predicted_token, text)})

    return {"output": result}

# @app.get("/")
# async def root():
#     return {"message": predict("today is a [mask]", top_k=5)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

