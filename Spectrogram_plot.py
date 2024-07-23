import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

def plot_spectrograms(input_dir):
 
    # Loop through each file 
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):  # Adjust the file extension (if needed)
            file_path = os.path.join(input_dir, filename)
            
            # Load audio 
            y, sr = librosa.load(file_path)

            # Compute the spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

            # Plot
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Logarithmic spectrogram - {filename}')
            plt.ylabel('Frequency [Hz]')  # Set y-axis label
            
            # Display plot 
            display(plt.gcf())
            plt.close()  # Close the plot to avoid memory issues
                      
          

