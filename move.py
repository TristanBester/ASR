import pandas as pd
import torchaudio
import os

df = pd.read_csv('prepared_csv/short_clips.csv')

counter = 0
for file in df['path']:
	if counter == 1000:
		break
	else:
		counter += 1

	try:
		path = os.path.join('short_clips', file)
		waveform, sample_rate = torchaudio.load(path)
		new_path = os.path.join('top_thou', file.split('.')[0] + '.wav')
		torchaudio.save(new_path, waveform, sample_rate)
	except:
		pass
