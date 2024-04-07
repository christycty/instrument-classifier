from __future__ import print_function

import os 
import pandas as pd
import numpy as np
import six
import soundfile as sf
import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

class featureExtractor():
    def __init__(self) -> None:
        pass
    
    def load_wav(self, wav_file):
        wav_data, sr = sf.read(wav_file, dtype='int16')
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    
        samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        
        # print(samples.shape, sr)
        # crop the first 3 seconds
        if len(samples) > sr*3:
            samples = samples[:sr*3]
            
        self.examples_batch = vggish_input.waveform_to_examples(samples, sr)        
        
        
        # self.examples_batch = vggish_input.wavfile_to_examples(wav_file)

    
    def extract_features(self):
        # Prepare a postprocessor to munge the model embeddings.
        pproc = vggish_postprocess.Postprocessor('code/vggish/vggish_pca_params.npz')
        
        with tf.Graph().as_default(), tf.Session() as sess:
            # Define the model in inference mode, load the checkpoint, and
            # locate input and output tensors.
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(sess, 'code/vggish/vggish_model.ckpt')
            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)
        
            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                        feed_dict={features_tensor: self.examples_batch})
            postprocessed_batch = pproc.postprocess(embedding_batch)
            self.postprocessed_batch = postprocessed_batch
            # print(postprocessed_batch)
            
        return postprocessed_batch
    
    def run(self, wav_files):
        postprocessed_batches = []
        for wav_file in wav_files:
            print("processing file:", wav_file)
            self.load_wav(wav_file)
            postprocessed_batches.append(self.extract_features())
        return postprocessed_batches
        

if __name__ == '__main__':
    # data_dir = 'data/irmas/IRMAS-Sample/Training'
    # data_dir = 'data/irmas/IRMAS-TrainingData'
    data_dir = 'data/MIS/raw/training'
    
    meta_file = 'data/MIS/raw/training.csv'
    
    # csv_dir = 'data/irmas/vggish/sample'
    # csv_dir = 'data/irmas/vggish/training'
    csv_dir = 'data/MIS'
    
#   tf.app.run()
    feature_extractor = featureExtractor()

    df = pd.read_csv(meta_file)
    classes = df['class'].unique()
    
    # drop_classes = ['gel', 'cel', 'flu', 'org', 'sax', 'tru']
    drop_classes = ['piano']
    
    for class_ in classes:
        if class_ in drop_classes:
            continue
        df_class = df[df['class'] == class_]
        
        print("processing class:", class_)
        
        file_list = df_class['filename'].tolist()
        file_list = [os.path.join(data_dir, file) for file in file_list]
        
        features = []
        postprocessed_batch = feature_extractor.run(file_list)
        for i, embedding in enumerate(postprocessed_batch):
            feature = {}
            feature['class'] = class_
            feature['filename'] = file_list[i]
            
            flatten_embedding = embedding.flatten()
            
            for j, emb in enumerate(flatten_embedding):
                feature[f'embedding_{j}'] = emb
            
            features.append(feature)
          
        df_feature = pd.DataFrame(features)
        csv_path = os.path.join(csv_dir, f'vggish-{class_}.csv')
        df_feature.to_csv(csv_path, index=False)
    
            
            
