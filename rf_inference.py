import os 
import sklearn
import numpy as np
import pandas as pd
import joblib
import librosa 

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns


dir = {
    'MIS_comb': {
        "model_dir": "code/models/misd_rf_comb",
        "csv_path": "data/MIS/combined-librosa-vggish.csv",
        "class_map": ['drum', 'guitar', 'piano', 'violin']
        },
    'MIS_vgg': {
        "model_dir": "code/models/misd_rf_comb",
        "csv_path": "data/MIS/combined-librosa-vggish.csv",
        "class_map": ['drum', 'guitar', 'piano', 'violin']
        },
    'IRMAS': {
        "model_dir": "code/models/irmas_rf_comb_10c",
        "csv_path": "data/IRMAS/combined-librosa-vggish.csv",
        "class_map": ['cel', 'cla', 'flu', 'gac', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']
        }

}

model_mode = 'MIS_comb'
model_dir = dir[model_mode]['model_dir']

demo_id = 3
demo_files = [
    ['ROOM_room1_MUS_bartok_DEV_redmi.wav', 'MIS_comb'],
    ['[vio][cla]2135__1.wav', 'IRMAS'],
    ['[pia][jaz_blu]1513__1.wav', 'IRMAS'],
    ['[voi][jaz_blu]2482__2.wav', 'IRMAS']
]
demo_file = demo_files[demo_id]

def load_features(csv_path):
    df = pd.read_csv(csv_path)

    # get row with filename
    row = df[df['filename'] == demo_file[0]]
    features = row.drop(columns=['class', 'filename'])
    return features

def run():
    print(f"Running inference for {demo_file[0]}...")
    print(f"Model mode: {model_mode}")
    
    rf = joblib.load(os.path.join(model_dir, 'rf.joblib')) 
    print(f"Loaded model from {model_dir}")
    
    features = load_features(dir[demo_file[1]]['csv_path'])
    print(f"Loaded features for {demo_file[0]}")
    
    pred = rf.predict(features)
    pred_class = dir[model_mode]['class_map'][pred[0]]
    
    print(f"Prediction: {pred_class}")
    print("*" * 50)
    
    
if __name__ == "__main__":
    run()