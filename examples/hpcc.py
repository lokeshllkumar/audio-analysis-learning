import librosa
import numpy as np

y, sr = librosa.load(librosa.ex('nutcracker'))

hop_length = 512

y_harmonic, y_percussive = librosa.effects.hpss(y)

tempo, beat_frames = librosa.beat.beat_track(y = y_percussive) # percussive elements are stronger indicators of rhythmic content 

mfcc = librosa.feature.mfcc(y = y, sr = sr, hop_length = hop_length, n_mfcc = 13) # MFCC algo is a typical characteristics extraction method

mfcc_delta = librosa.feature.delta(mfcc) # a type of transformation

beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames) # another type of transformation

chromagram = librosa.feature.chroma_cqt(y = y_harmonic, sr = sr)

beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate = np.median)

beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

print(beat_features)