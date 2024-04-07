import os
import cv2
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import extract_features


# https://github.com/cosmir/openmic-2018/blob/durand/baseline/scripts/Baseline.ipynb
class openmicProcessor:
    def __init__(self) -> None:
        self.class_to_index = {
            "accordion": 0,
            "banjo": 1,
            "bass": 2,
            "cello": 3,
            "clarinet": 4,
            "cymbals": 5,
            "drums": 6,
            "flute": 7,
            "guitar": 8,
            "mallet_percussion": 9,
            "mandolin": 10,
            "organ": 11,
            "piano": 12,
            "saxophone": 13,
            "synthesizer": 14,
            "trombone": 15,
            "trumpet": 16,
            "ukulele": 17,
            "violin": 18,
            "voice": 19,
        }
        
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    def load_npz(self, npz_file):
        OPENMIC = np.load(npz_file, allow_pickle=True)

        X, Y_true, Y_mask, sample_key = (
            OPENMIC["X"],
            OPENMIC["Y_true"],
            OPENMIC["Y_mask"],
            OPENMIC["sample_key"],
        )

        return X, Y_true, Y_mask, sample_key

    def extract_3sec(self, X):
        # X from (, 10, ) to (, 3, ) by taking the first 3 seconds
        X_3sec = X[::3, :]
        return X_3sec

    def filter_classes(self, X, Y_true, Y_mask, sample_key, classes):
        # filter classes
        class_indices = [i for i, class_ in enumerate(sample_key) if class_ in classes]

    def stat(self, Y_mask, Y_true):
        # count average numbers of true in Y_mask
        print(f"avg label count {np.mean(np.sum(Y_mask, axis=1))}")
        
        # count data with three class labels or above
        # number of rows with at least 3 True values
        print(f"number of rows with at least 3 True values {np.sum(np.sum(Y_mask, axis=1) >= 3)}")
        
        stat = []
        # count pair of indices that co-appears in Y_mask
        for i in range(0, 19):
            for j in range(i + 1, 20):
                count = np.sum(Y_mask[:, i] & Y_mask[:, j])
                # print(f'Indices {i} and {j} are both true: {count}')

                stat.append((self.index_to_class[i], self.index_to_class[j], count))

        # convert to df
        stat_df = pd.DataFrame(stat, columns=["i", "j", "count"])
        # print highest 10 counts w class name
        print(stat_df.sort_values("count", ascending=False).head(15))
        
        # Iterate through all combinations of three indices
        stat_three = []
        for i in range(0, 18):
            for j in range(i + 1, 19):
                for k in range(j + 1, 20):
                    indices = [i, j, k]
                    
                    # all_labels_count = 0
                    # for l in range(Y_mask.shape[0]):
                    #     if np.all(Y_mask[l, indices]):
                    #         # print(f"Indices {i}, {j}, {k} are all true in row {l}")
                    #         if Y_true[l, i] > 0.5 and Y_true[l, j] > 0.5 and Y_true[l, k] > 0.5:
                    #             all_labels_count += 1
                            
                    
                    # count rows where all three indices are true and Y_true > 0.5
                    rows_with_all_labels = np.all(Y_mask[:, indices], axis=1)
                    rows_with_threshold = Y_true[rows_with_all_labels] > 0.5
                    
                    # count all rows in rows_with_threshold with at least 3 true values
                    true_count = np.sum(rows_with_threshold, axis=1)
                    # count total number of values in true_count > 3
                    all_labels_count = np.sum(true_count >= 3)
                                        
                    # convert indices to class names
                    indices = [self.index_to_class[i] for i in indices]
                    stat_three.append((indices, all_labels_count))
        
        stat_three_df = pd.DataFrame(stat_three, columns=["indices", "count"])
        print(stat_three_df.sort_values("count", ascending=False).head(15))
        
        indices = [13, 15, 16]
        for l in range(Y_mask.shape[0]):
            if np.all(Y_mask[l, indices]):
                # print(f"Indices {i}, {j}, {k} are all true in row {l}")
                if Y_true[l, 13] > 0.5 and Y_true[l, 15] > 0.5 and Y_true[l, 16] > 0.5:
                    all_labels_count += 1
        print(all_labels_count)

    def extract_vggish(self):
        X, Y_true, Y_mask, sample_key = self.load_npz(
            "data/openmic-raw/openmic-2018.npz"
        )
        print(X.shape, Y_true.shape, Y_mask.shape, sample_key.shape)

        # print first data
        print(Y_true[0], Y_mask[0], sample_key[0])

        self.stat(Y_mask, Y_true)

        for i in range(10):
            Y_fil = Y_true[i][Y_mask[i]]
            # index of True values in Y_mask[i]
            index = np.where(Y_mask[i])[0].tolist()
            print(index, Y_fil, sample_key[i])

        X_trimmed = self.extract_3sec(X)
        print(X_trimmed.shape)
        
        


if __name__ == "__main__":
    processor = openmicProcessor()
    processor.extract_vggish()
