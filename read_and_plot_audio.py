import os
import librosa
import matplotlib.pyplot as plt
import numpy as np



# Function to read and plot all audio files in the specified folder
def read_and_plot_audio_files(folder_path):
    
    p_ref = 20e-6  #Reference sound pressure level (in Pascals).
    amplitudes_dict = {} #Save amplitudes

    # Make a list of all files in the folder:
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):  #Process only .wav files in the folder
            file_path = os.path.join(folder_path, file_name)
            audio, sr = librosa.load(file_path, sr=None)
            
            # Normalize the audio signal to Pascal
            audio = audio / np.max(np.abs(audio)) * p_ref
            
            # Create a time array in seconds
            duration = len(audio) / sr
            time = np.linspace(0., duration, len(audio))
            
            # Plot
            plt.figure(figsize=(10, 4))
            plt.plot(time, audio, label='Waveform')
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [Pa]')
            plt.title(f'Time vs Amplitude for {file_name}')
            plt.legend()
            plt.show()
                      
            # Store amplitude data to calculate kurtosis later
            amplitudes_dict[file_name] = audio
    

          
