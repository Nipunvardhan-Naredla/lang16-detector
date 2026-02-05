import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import unicodedata

#Tokenizer-I decided to save it locally to prevent confusion later on
#It is bert-base-multilingual-cased

#uses cuda if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("model/tokenizer")

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

    #sets up the model
    model = LanguageModel()
    print(model)

    #Loss Function
    criterion = nn.CrossEntropyLoss()

    #Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Sets up datasets
    #batch 32 works good, i think
        
    train_dataset = LanguageDataset("completed_datasets/train_data.json")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = LanguageDataset("completed_datasets/val_data.json")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    test_dataset = LanguageDataset("completed_datasets/test_data.json")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


    '''This is to test overfitting, to see if the model works
    train_dataset = LanguageDataset("completed_datasets/very_small_test.json")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    val_dataset = LanguageDataset("completed_datasets/very_small_test.json")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    test_dataset = LanguageDataset("completed_datasets/very_small_test.json")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    '''
    
    #Training Loop
    max_epochs = 3

    train_loss_list = []
    val_loss_list = []
    accuracy_list = []
    final_epoch = 0
    
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        
        #training phase
        model.train()
        running_loss = 0.0
        for data, target in  tqdm(train_loader, desc="Training Loop"):
            #data = data["input_ids"].float()
            target = target.long()
            optimizer.zero_grad()#clears gradients
            outputs = model(data)#forward pass
            loss = criterion(outputs, target)#calculates loss
            loss.backward()#backward pass
            optimizer.step()#updates model
            
            running_loss += loss.item() * target.size(0)
            
        train_loss = running_loss / len(train_loader.dataset)
        train_loss_list.append(train_loss)
        
        #validation phase
        model.eval()
        running_loss = 0.0
        total_accurate = 0
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc = "Validation Loop"):
                #data = data['input_ids'].float()
                target = target.long()
                #target = target['input_ids'].float()
                outputs = model(data)

                preds = torch.argmax(outputs, dim=1)
                
                total_accurate += (preds == target).sum().item()


                loss = criterion(outputs, target)
                running_loss += loss.item() * target.size(0)
            accuracy = total_accurate/len(val_loader.dataset)*100
                
        val_loss = running_loss / len(val_loader.dataset)
            

        val_loss_list.append(val_loss)
        accuracy_list.append(accuracy)
        final_epoch = epoch


        #print epoch data
        print(f"Epoch {epoch+1}/{max_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}, Accuracy: {accuracy}\n")

        #saves training data to file
        info_dict = {"Training_loss": train_loss_list,
                     "Validation_loss":  val_loss_list,
                     "Accuracy": accuracy_list,
                     "Epochs": max_epochs,
                     "Epoch_Reached":final_epoch,}
        with open("model/info.json", "w", encoding="utf-8") as file:
            json.dump(info_dict, file, indent=4)
        
        #Checks if accuracy is really good
        if accuracy >= 98:
            break
        
        #patience
        if epoch > 4:#makes sure it isn't first four epochs
            patience = set(accuracy_list[-4:])
            if max(patience)-min(patience) <= 0.3: #means in the last four epochs nothing changed
                print("\nPeak Accuracy Achevieved")
                break
        
    
    #tests accuracy
    model.eval()
    print("\nTesting phase")
    
    total_accurate = 0
    for data, target in tqdm(test_loader, desc = "Test Loop"):
        #data = data['input_ids'].float()
        target = target.long()

        outputs = model(data)

        preds = torch.argmax(outputs, dim=1)
                
        total_accurate += (preds == target).sum().item()
        
    print(f"\n{float(total_accurate)/float(len(test_loader))*100}% Accurate")

    print("\nSaving Phase")
    
    #Saves model

    torch.save(model.state_dict(), "model/language_detection_model.pth")
    
    print("Complete")
            
    
