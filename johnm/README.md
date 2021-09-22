# MLB Model

These notes pertain mostly to code used for training the model, which resides in this repo. Please see https://www.kaggle.com/brandenkmurray/mlb-predict-final?scriptVersionId=69500207 for the code used during inference.


## Instructions
1. Create pickle files using the notebook at https://www.kaggle.com/naotaka1128/creating-unnested-dataset/notebook. The code here uses them instead of the raw csv files.
2. Install required libraries from requirements.txt
3. Run mlb_dataprep_24f.py to create the dataset used for training. 
4. Run mlb_train_24f.py to train a model on the dataset.
5. Use the serialized models output from the above step as inputs to the submitted notebook.
6. Get 2nd place!


## System
- AMD Ryzen Threadripper 3970X 32-Core Processor
- 256GB RAM DDR4
- Quadro RTX 6000 24GB (3)
- Ubuntu 18.04
- Python 3.9.7
