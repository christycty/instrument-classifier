import librosa

def extract_features(audio_path, duration=None):
    if duration:
        y, sr = librosa.load(audio_path, duration=duration)
    else:
        y, sr = librosa.load(audio_path)  
    
    features = {}
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    for i in range(10):
        features[f'MFCC_{i+1}_mean'] = mfccs[i].mean()
        features[f'MFCC_{i+1}_var'] = mfccs[i].var()
        
        
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['SpectralCentroid_mean'] = centroid.mean()
    features['SpectralCentroid_var'] = centroid.var()
    
    flux = librosa.onset.onset_strength(y=y, sr=sr)
    features['SpectralFlux_mean'] = flux.mean()
    features['SpectralFlux_var'] = flux.var()
    
    zcr = librosa.feature.zero_crossing_rate(y)  
    features['ZeroCrossingRate_mean'] = zcr.mean()
    features['ZeroCrossingRate_var'] = zcr.var()
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr) 
    features['SpectralRollOff_mean'] = rolloff.mean()
    features['SpectralRollOff_var'] = rolloff.var()
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(10):
        features[f'Chroma_{i+1}_mean'] = chroma[i].mean()
        features[f'Chroma_{i+1}_var'] = chroma[i].var()
    
    return features
