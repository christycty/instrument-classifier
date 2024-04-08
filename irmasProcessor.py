import os 
import cv2 
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import extract_features


# all the processing for the IRMAS dataset
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

    # generate spectrogram for each audio
    def gen_spectrogram(self, save_dir):
        
        classes = os.listdir(self.wav_dir)
        try:
            classes.remove("gel")
        except:
            pass 
            
        i = 0
        for class_ in classes:
            print("processing class:", class_)
            
            # create dir
            os.makedirs(os.path.join(save_dir, class_), exist_ok=True)
            class_dir = os.path.join(self.wav_dir, class_)
            
            for wav_file in os.listdir(class_dir):
                # print(f"processing file {i}:", wav_file)
                # i += 1
                # if i <= 3482:
                #     continue
                
                wav_path = os.path.join(class_dir, wav_file)
                y, sr = librosa.load(wav_path)
                S = librosa.feature.melspectrogram(y=y, sr=sr)
                S_dB = librosa.power_to_db(S, ref=np.max)
                img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, cmap='viridis')
                # remove border 
                plt.axis('off')
                plt.tight_layout(pad=0)
                # save img to path
                img_path = os.path.join(save_dir, class_, wav_file[:-4] + '.png')
                plt.savefig(img_path)
                plt.clf()
    
    # rename the generated spectrogram for upload to kaggle
    def spectro_rename(self, dir):
        classes = os.listdir(dir)
        for class_ in classes:
            for filepath in os.listdir(os.path.join(dir, class_)):
                # change all [] to _ in filename
                new_filepath = filepath.replace('[', '_').replace(']', '_')
                os.rename(os.path.join(dir, class_, filepath), os.path.join(dir, class_, new_filepath))
    
    # merge the vggish features into one csv file
    def merge_vgg(self, vgg_dir, csv_name):
        files = os.listdir(vgg_dir)
        
        for i, file in enumerate(files):
            if i == 0:
                df = pd.read_csv(os.path.join(vgg_dir, file))
            else:
                df = pd.concat([df, pd.read_csv(os.path.join(vgg_dir, file))])
        
        csv_path = os.path.join(vgg_dir, csv_name)
        df.to_csv(csv_path, index=False)
    
    def filter_csv(self, csv_old_path, csv_new_path, classes):
        df = pd.read_csv(csv_old_path)
        df = df[df['class'].isin(classes)]
        df.to_csv(csv_new_path, index=False)
        return df
        
        
if __name__ == "__main__":
    vgg_dir = 'data/irmas/vggish/training'
    # csv_name = 'data/irmas/IRMAS-Training.csv'
    # csv_new_name = 'data/irmas/IRMAS-Training-5Class.csv'
    
    csv_name = 'data/irmas/combined-librosa-vggish.csv'
    csv_new_name = 'data/irmas/combined-librosa-vggish-5class.csv'
    
    # wav_dir = 'data/irmas/IRMAS-Sample/Training'
    wav_dir = 'data/irmas/IRMAS-TrainingData'
    processor = irmasProcessor(wav_dir)
    # processor.merge_vgg(vgg_dir, csv_name)
    
    processor.filter_csv(csv_name, csv_new_name, ['cla', 'gac', 'pia', 'tru', 'voi'])
    exit()
    spectro_dir = 'data/irmas/spectrogram/training_veridis'
    # spectro_dir = 'data/irmas/spectrogram/sample'
    # spectro_dir = 'data/irmas/spectrogram/training'
    
    processor.gen_spectrogram(spectro_dir)
    
    processor.spectro_rename(spectro_dir)
    
    # df = processor.gen_librosa_features()

    # csv_path = 'data/irmas/IRMAS-Training-5Class.csv'
    # df.to_csv(csv_path, index=False)