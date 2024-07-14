# from transformers import CLIPProcessor, CLIPModel
# import sys
# import json

# # Load CLIP model and processor
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# # openai/clip-vit-base-patch32

# def get_text_features(text):
#     inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
#     query_embedding = clip_model.get_text_features(**inputs).squeeze().detach().numpy().tolist()
#     return query_embedding

# if __name__ == "__main__":
#     try:
#         query_text = sys.argv[1]
#         embedding = get_text_features(query_text)
#         print(json.dumps(embedding))
#     except Exception as e:
#         print(json.dumps({"error": str(e)}), file=sys.stderr)
# ml_script.py
# process_text.py
from transformers import CLIPProcessor, CLIPModel, pipeline
import sys
import json

import torch
print(torch.__version__)

# Load CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_text_features(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
    query_embedding = clip_model.get_text_features(**inputs).squeeze().detach().numpy().tolist()
    return query_embedding

def answer_question(question, context):
    qa_model = pipeline('question-answering', model='deepset/roberta-base-squad2')
    answer = qa_model(question=question, context=context)
    return answer['answer']

if __name__ == "__main__":
    try:
        task = sys.argv[1]
        if task == "embed":
            query_text = sys.argv[2]
            embedding = get_text_features(query_text)
            print(json.dumps(embedding))
        elif task == "qa":
            question = sys.argv[2]
            context = sys.argv[3]
            answer = answer_question(question, context)
            print(json.dumps(answer))  # Ensure the answer is also JSON encoded
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
