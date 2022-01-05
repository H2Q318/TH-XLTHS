from os.path import join
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH_NAHL = r"D:\Documents\XLTHS\Thi TH\NguyenAmHuanLuyen-16k"
FILE_PATH_WAV_NAHL = ['01MDA', '02FVA', '03MAB', '04MHB']
FILE_WAV_NAHL = ['a.wav', 'e.wav', 'i.wav', 'o.wav', 'u.wav']

INDEX = 0
FRAME_LENGTH = 0.03
OVERLAP_FRAME = 0.02

# Main
for i in range(0, len(FILE_PATH_WAV_NAHL)):
  signal = []
  for j in range(0, len(FILE_WAV_NAHL)):    
    # Đọc dữ liệu đầu vào
    frequency, signalVowel = read(join(FILE_PATH_NAHL, FILE_PATH_WAV_NAHL[i], FILE_WAV_NAHL[j]))
    signal.extend(signalVowel)
  signal = signal / max(np.max(signal), abs(np.min(signal)))
  frameLength = int(FRAME_LENGTH * frequency) # Độ dài của 1 frame (đơn vị mẫu)
  overlapFrame = int(OVERLAP_FRAME * frequency)
  
  # Tín hiệu trên miền thời gian
  timeSample = np.zeros(len(signal))
  for index in range(len(signal)):
    timeSample[index] = index / frequency
  
  # Show
  plt.figure(i + 1)
  plt.subplot(2, 1, 1)
  plt.title(f"Signal: {FILE_PATH_WAV_NAHL[i]}")
  plt.plot(timeSample, signal)
  plt.xlabel('Time(s)')
  plt.ylabel('Signal Amplitude')
  plt.xlim([0, len(signal) / frequency])
    
  plt.subplot(2, 1, 2)
  plt.title(f"Wideband Of Signal: {FILE_PATH_WAV_NAHL[i]}")
  plt.specgram(signal, NFFT=frameLength, Fs=frequency, window=np.hamming(frameLength), noverlap=overlapFrame, cmap='jet')
  plt.xlabel('Time(s)')
  plt.ylabel('Frequence(Hz)')
  plt.tight_layout() 
plt.show()