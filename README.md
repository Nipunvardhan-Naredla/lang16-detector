# lang16-detector
  A lightweight pytorch model that can classify 16 different languages

 ### Table of Contents
- Intro
- Model
- Datasets 
- Training
- Possible Errors
- Future
- Notes

---

### Intro
  lang16-detector is a lightweight pytorch model, that can classify these 16 languages:
  - Catalan
  - Danish
  - German
  - Modern Greek
  - English
  - Esperanto
  - Spanish
  - Finnish
  - French
  - Hungarian
  - Italian
  - Latin
  - Dutch
  - Portugese
  - Swedish
  - Chinese

---

### Model
The model is around 4 million parameters, with around 8 layers: 1 embedding, 6 LSTM layers, and 1 linear. It uses bert base multilingiual uncased as the tokenizer. It acheived around 92% accuracy on the val_data dataset and 93% on the test_data dataset.

---

### DataSets
This repository contains 5 datasets:
  - large_test_data.json ~ 24 million tokens
  - test_data.json ~ 2.4 million tokens
  - train_data.json ~ 19.2 million tokens
  - val_data.json ~ 2.4 million tokens
  - very_small_test.json ~ 200 tokens


All of these datasets were created from files obtained from Project Gutenberg. They were sorted via guten-sort(another one of my public repositories) and a few helper scripts, which I included in this repository under dataset creation. These helper scripts are not documented as well as the training script and the querying script, and are most likely not optimized at all. But they work and should allow someone to recreate these datasets if neccesary.

The datasets are evenly split with no class taking up more than 7% of the total datasets. When using these datasets, the user will not need to worry about uneven distribution.

---

### Training
Training took 3 epochs with each epoch taking around 27 minutes. With more power, a user could shorten the training time. Then again this was trained on a laptop 4060, so if a user wanted to edit the model architecture, it shouldn't take too long to retrain. However, the model achieves around 90% accuracy already, so unless exxtreme accuracy is desired, I wouldn't change it. 

---

### GUI for lang16 detector
There is a folder in this repository named "GUI for lang16-detector". I used customtkinter to build a simple GUI that allows a user to type into a textbox and click enter. The GUI will display the language and the confindence in that language. To use the gui, install the requirements.txt and run window.pyw. 

The UI:
![A picture of the UI](/GUI_picture.png)

---

### Using in your own projects:

1.  Copy the lang16_detector in the "GUI for lang16-detector" folder into your project.
2.  Import the function into your script with this line: ```from lang16_detector import query_model```
3.  query_model has 1 argument- the string you want to know the language of.
4.  query_model will return a dictonary with 2 keys: "class" and "confidence". Class is the language the model thinks the string is and confidence is the confidence in that class

---

### Possible Errors

- Because the model was trained on certain books, the model might distinguish one word to be part of a language. A specific noun may cause the model to choose the wrong class but there should theoretically be only a couple nouns like that.
- Due to being trained on public domain texts that are older, it might get current slang incorrect. 

---

### Future
In the event I decide to revisit this project, I will try to include all the languages in the tokenizer, bert base multilingual uncased. I would also try to shrink the model to a smaller size.

---

### Notes
- Due to the datasets being quite large, I added them to the releases.


