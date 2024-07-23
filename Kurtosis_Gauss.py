import os
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import kurtosis

def kurtosis_plus_gauss(folderPath, excel_path, newpath, lowerdB=80, upperdB=120, d_s=1, s_r=44100):
   
    os.makedirs(newpath, exist_ok=True)

    # Leer los audios
    audioFiles = [f for f in os.listdir(folderPath) if f.endswith('.wav')]
    L = len(audioFiles)
    list_dB = np.arange(lowerdB, upperdB + 1, 5)
    list_dB_full = ["Original"] + list(list_dB)

    # Initialize storage for audio data and kurtosis values
    audioK = np.zeros((L, len(list_dB) + 1))

    # Read Lpeak values from the Excel file
    LpeakData = pd.read_excel(excel_path, usecols='C', skiprows=2, nrows=22)
    Lpeak = LpeakData.values.flatten()

    p_ref = 20e-6  # Reference sound pressure level in Pascals

    # Process each audio file
    for a, name in enumerate(audioFiles):
        audio, fs = sf.read(os.path.join(folderPath, name))

        if a < len(Lpeak):
            Lpeaki = Lpeak[a]
            p_peak = p_ref * 10 ** (Lpeaki / 20)

            # Scaling
            p = audio * (p_peak / np.max(audio))
            l = len(p)
            t = np.arange(0, l / fs, 1 / fs)

            # Calculate kurtosis
            k = kurtosis(p)
            audioK[a, 0] = k

            for i, dBi in enumerate(list_dB):
                Leq_linear = 2e-5 * 10 ** (dBi / 20)  # Convert dB to linear scale
                sigma = Leq_linear  # Standard deviation of Gaussian distribution
                num_samp = d_s * s_r  # Number of samples
                gauss_noise = sigma * np.random.randn(num_samp)  # Generate Gaussian noise

                # Add signals
                gnI = p + gauss_noise

                ki = kurtosis(gnI)  # Kurtosis of the new signal
                audioK[a, i + 1] = ki

                # Normalize
                maxp = np.max(np.abs(gnI))
                Lpeak_new = 10 * np.log10((maxp ** 2) / (p_ref ** 2))
                gnI = gnI / maxp

                # Save the new audio file
                baseName, _ = os.path.splitext(name)
                filename = f"{baseName}_{dBi}_{Lpeak_new:.2f}.wav"
                sf.write(os.path.join(newpath, filename), gnI, fs)

                list_dB_full[i + 1] = dBi

    # Create table
    Tablek = pd.DataFrame(audioK, index=audioFiles, columns=list_dB_full)

    return Tablek
