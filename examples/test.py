import librosa 

y, sr = librosa.load(librosa.ex('nutcracker'))
tempo, beat_frames = librosa.beat.beat_track(y = y, sr = sr)

print(f'Tempo: {tempo}')
beat_times = librosa.frames_to_time(beat_frames, sr = sr)

print(beat_times)