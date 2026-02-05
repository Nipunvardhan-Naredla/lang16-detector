from transformers import AutoTokenizer
from model import LanguageModel
import torch
import json
import sys


input_string = "Bonjour, comment allez-vous aujourd'hui ?"

if __name__ == "__main__":
    #get input_string from argument
    input_string = sys.argv[1]

    
    #tokenizer
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")

    #This gets the dictonary of the classes
    with open("classes.json") as file:
        classes_dict = json.load(file)

    #tokenizes data
    tokenized_data = tokenizer(input_string,
                               return_tensors="pt",
                               padding="max_length",
                               max_length=15,
                               truncation=True,
                               )['input_ids'].long()

    #loads model
    model = LanguageModel()

    #print(model)
    model.load_state_dict(torch.load('language_detection_model.pth', weights_only=True))
    
    #sets model to eval 
    model.eval()

    #passes it through model
    outputs = model(tokenized_data)

    preds = torch.argmax(outputs, dim=1).sum().item()

    guess = classes_dict[str(preds)]
    print(guess)
