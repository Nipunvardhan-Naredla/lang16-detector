from transformers import AutoTokenizer
import torch.nn.functional as F
#from lang16_detector import LanguageModel
import torch
import json
import sys
import os

def query_model(input_string):
    '''querys the model and returns a dictonary

    Arguments:
        input_string: prompt for model

    Returns:
        guess_dict: a dictonary with class and confidence in the class
    '''
    #changes the directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
    from .model import LanguageModel
    model = LanguageModel()

    #print(model)
    model.load_state_dict(torch.load('language_detection_model.pth', weights_only=True))
    
    #sets model to eval 
    model.eval()

    #passes it through model
    outputs = model(tokenized_data)
    probs = F.softmax(outputs, dim=1)
    
    pred_class = torch.argmax(probs, dim=1).sum().item()  # predicted class
    
    confidence = probs.max(dim=1).values.item()

    confidence = float(str(confidence * 100)[:5])

    #returns the dictonary
    return {"class": classes_dict[str(pred_class)], "confidence": confidence}


if __name__ == "__main__":
    #input_string = "Cookies are better than cakes if they don't have raisins"
    print(query_model(sys.argv[1]))
