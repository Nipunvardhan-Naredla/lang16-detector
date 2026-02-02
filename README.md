# lang16-detector
  A lightweight pytorch model that can classify 16 different languages 

 ### Table of Contents

- Intro
- Datasets 
- Training

---

### Intro
  lang16-detector is a extermly lightweight pytorch model, that can classify these 16 languages:
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

### DataSets
This repository contains 5 datasets:
  - large_test_data.json ~ 30 million tokens
  - test_data.json ~ 3 million tokens
  - train_data.json ~ 24 million tokens
  - val_data.json ~ 3 million tokens
  - very_small_test.json ~ 200 tokens


All of these datasets were created from files obtained from Project Gutenberg. They were sorted via guten-sort(another one of my public repositories) and a few helper scripts, which I included in this repository under dataset creation. These helper scripts are not documeted as well as the training script and the querying script, and are most likely not optimized at all. But they do what they have to and should allow someeone to recreate these datasets if neccesary





