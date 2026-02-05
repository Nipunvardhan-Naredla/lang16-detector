import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import unicodedata

#switches it to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#filepaths
test_json = "completed_datasets/val_data.json"
model_path = "model/language_detection_model.pth"
tokenizer_path = "model/tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

#classes
#This gets the dictonary of the classes
with open("completed_datasets/classes.json") as file:
    classes_dict = json.load(file)

#Dataset Class
class LanguageDataset(Dataset):
    def __init__(self, json_file_path):
        '''Intializes the dataset

        Arguments:
            json_file_path: file path to json file
        '''
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        data_list = list(data.items())#turns dictonary into list of pairs

        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''accesing items

        Arguments:
            idx: index of item to return

        Returns:
            A tuple with the tokenized data and the target
        '''
        item = self.data[idx]

        input_data = item[0]
        target = classes_dict[item[1]]
        input_data = unicodedata.normalize("NFKC", input_data)

        #Makes the model input 15 tokens no matter what
        tokenized_data = tokenizer(input_data,
                                   return_tensors="pt",
                                   padding="max_length",
                                   max_length=15,
                                   truncation=True,
                                   )['input_ids'].long()
        return tokenized_data, target

#model class
class LanguageModel(nn.Module):
    def __init__(self):
        '''Defines model architecture


        '''
        super().__init__()
        self.embedding = nn.Embedding(120000, 15)
        
        self.relu = nn.ReLU()
        
        self.fc1 = nn.LSTM(input_size = 15,
                          hidden_size = 240,
                          num_layers = 6,
                          batch_first = True,
                          )

        self.fc2 = nn.Linear(240,16)#change 16 depending on amount of classes

    def forward(self, x):
        '''the forward pass
        '''
        x = self.embedding(x).long()
        x = x.squeeze(1).float()
        x, _ = self.fc1(x)
        x = x[:, -1, :]
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    #loads data
    test_dataset = LanguageDataset(test_json)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    #loads model
    model = LanguageModel()
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()

    total_accurate = 0
    for data, target in tqdm(test_loader, desc = "Test Loop"):
        target = target.long()

        outputs = model(data)

        preds = torch.argmax(outputs, dim=1)
                
        total_accurate += (preds == target).sum().item()
        
    print(f"\n{float(total_accurate)/float(len(test_loader))*100}% Accurate")




















    
