import os 
import cv2 
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import extract_features
import shutil


# all the processing for the MISD dataset
class misdProcessor():
    def __init__(self, wav_dir, meta_file) -> None:
        self.wav_dir = wav_dir
        self.meta_file = meta_file
    
    # generate librosa features for all the audio files in the dataset
    def gen_librosa_features(self):  
        df_meta = pd.read_csv(meta_file)
        classes = df_meta["class"].unique()
    
        data = []
        
        for class_ in classes:
            df_class = df_meta[df_meta["class"] == class_]
            
            print("processing class:", class_)
            for wav_file in df_class["filename"]:
                wav_path = os.path.join(self.wav_dir, wav_file)
                features = extract_features(wav_path, duration=3)
                features['class'] = class_
                features['filename'] = wav_file
                data.append(features)
        
        return pd.DataFrame(data)

    # generate spectrogram for each audio
    def gen_spectrogram(self, save_dir):
        
        # classes = os.listdir(self.wav_dir)
        classes = ['violin']
        df_meta = pd.read_csv(meta_file)
        
        i = 0
        for class_ in classes:
            df_class = df_meta[df_meta["class"] == class_]
            
            # create dir
            os.makedirs(os.path.join(save_dir, class_), exist_ok=True)
            class_dir = os.path.join(self.wav_dir, class_)
            
            for wav_file in df_class["filename"]:
                wav_path = os.path.join(self.wav_dir, wav_file)
                
                y, sr = librosa.load(wav_path, duration=3)
                
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
        
    # put filenames into dataframe
    def list_files(self):
        files = os.listdir(self.wav_dir)
        df = pd.DataFrame(files, columns=['filename'])
        df.to_csv('data/MIS/raw/training.csv', index=False)
        
    def move_files(self, spec_old_dir, spec_new_dir):
        classes = os.listdir(spec_new_dir)
        # mark which class folder each file belongs to
        class_dict = {}
        for class_ in classes:
            for file in os.listdir(os.path.join(spec_new_dir, class_)):
                class_dict[file[:-4]] = class_
        
        unlabelled = []
        df = pd.read_csv(self.meta_file)
        for i, row in df.iterrows():
            class_ = row['class']
            filename = row['filename'][:-4]
            if filename not in class_dict:
                print(f"File {filename} not found in spectrogram folder")
                unlabelled.append(filename)
                continue
            
            # check if file in correct class folder
            if class_dict[filename] != class_:
                print(f"File {filename} in wrong class folder")
                # move file to correct class folder
                shutil.copy(os.path.join(spec_old_dir, class_dict[filename], filename + '.png'), os.path.join(spec_new_dir, class_, filename + '.png'))
        
        # store unlabelled files to csv
        df_unlabelled = pd.DataFrame(unlabelled, columns=['filename'])
        df_unlabelled.to_csv('data/MIS/unlabelled.csv', index=False)
    
    def move_(self, spec_dir):
        # mark which class folder each file belongs to
        df = pd.read_csv(self.meta_file)
        
        class_dict = {}
        for i, row in df.iterrows():
            class_dict[row['filename'][:-4]] = row['class']
        
        
        classes = os.listdir(spec_dir)
        for class_ in classes:
            for file in os.listdir(os.path.join(spec_dir, class_)):
                if file[:-4] not in class_dict:
                    print(f"File {file} not found in meta file")
                    os.remove(os.path.join(spec_dir, class_, file))
                    continue
                
                if class_dict[file[:-4]] != class_:
                    print(f"File {file} in wrong class folder")
                    os.remove(os.path.join(spec_dir, class_, file))
                    
        
if __name__ == "__main__":
    vgg_dir = 'data/irmas/vggish/training'
    csv_name = 'vggish-all.csv'
    
    # wav_dir = 'data/irmas/IRMAS-Sample/Training'
    # wav_dir = 'data/irmas/IRMAS-TrainingData'
    wav_dir = 'data/MIS/raw/training'
    meta_file = 'data/MIS/raw/training.csv'
    processor = misdProcessor(wav_dir, meta_file)
    
    # processor.list_files()
    # processor.merge_vgg(vgg_dir, csv_name)
    
    
    # spectro_dir = 'data/irmas/spectrogram/training_veridis'
    # spectro_dir = 'data/irmas/spectrogram/sample'
    # spectro_dir = 'data/irmas/spectrogram/training'
    spectro_dir = 'data/MIS/training_trim'
    spectro_old_dir = 'data/MIS/training_trim_old'
    
    # processor.move_files(spectro_old_dir, spectro_dir)
    # processor.remove_wrong_files(spectro_dir)
    # exit()
    processor.gen_spectrogram(spectro_dir)
    # processor.spectro_rename(spectro_dir)
    
    # df = processor.gen_librosa_features()

    # csv_path = 'data/mis/librosa-feature.csv'
    # df.to_csv(csv_path, index=False)