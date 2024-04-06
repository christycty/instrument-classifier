import os 
import cv2 
import librosa
import numpy as np
import pandas as pd

from utils import extract_features

class irmasProcessor():
    def __init__(self, wav_dir) -> None:
        self.wav_dir = wav_dir
    
    # generate librosa features for all the audio files in the dataset
    def gen_librosa_features(self):
        classes = os.listdir(self.wav_dir)
        drop_classes = ['gel', 'cel', 'flu', 'org', 'sax', 'tru']
        # remove drop_classes from classes
        classes = [class_ for class_ in classes if class_ not in drop_classes]
        
        data = []
        
        for class_ in classes:
            print("processing class:", class_)
            class_dir = os.path.join(self.wav_dir, class_)
            for wav_file in os.listdir(class_dir):
                wav_path = os.path.join(class_dir, wav_file)
                features = extract_features(wav_path)
                features['class'] = class_
                features['filename'] = wav_file
                data.append(features)
        
        return pd.DataFrame(data)
    
if __name__ == "__main__":
    wav_dir = 'data/irmas/IRMAS-TrainingData'
    processor = irmasProcessor(wav_dir)
    
    df = processor.gen_librosa_features()

    csv_path = 'data/irmas/IRMAS-Training-5Class.csv'
    df.to_csv(csv_path, index=False)