from os.path import join
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import frame
import ste
import pitch

FILE_PATH_THHL = 'D:\Documents\XLTHS\Thi TH\TinHieuHuanLuyen-44k'
FILE_WAV_THHL = ['01MDA.wav', '02FVA.wav', '03MAB.wav', '06FTB.wav']
FILE_LAB_THHL = ['01MDA.lab', '02FVA.lab', '03MAB.lab', '06FTB.lab']

LINE = [[0.45, 0.81, 1.53, 1.85, 2.69, 2.86, 3.78, 4.15, 4.84, 5.14], 
        [0.83, 1.37, 2.09, 2.60, 3.57, 4.00, 4.76, 5.33, 6.18, 6.68], 
        [1.03, 1.42, 2.46, 2.80, 4.21, 4.52, 6.81, 7.14, 8.22, 8.50], 
        [1.52, 1.92, 3.91, 4.35, 6.18, 6.60, 8.67, 9.14, 10.94, 11.33]]

INDEX = 0
TIME_FRAME = 0.03

# Main
for i in range(0, len(FILE_WAV_THHL)):
  # Đọc dữ liệu đầu vào
  frequency, signal = read(join(FILE_PATH_THHL, FILE_WAV_THHL[i]))
  signal = signal / max(np.max(signal), abs(np.min(signal)))
  frameLength = int(TIME_FRAME * frequency) # Độ dài của 1 frame (đơn vị mẫu)
  framesArray = frame.getFramesArray(signal, frameLength)
  STEArray = ste.calSTE(framesArray)
  markPoint = ste.findSpeechAndSilence(STEArray)
  
  # Tín hiệu trên miền thời gian
  timeSample = np.zeros(len(signal))
  for index in range(len(signal)):
    timeSample[index] = index / frequency
  
  timeSampleSTE = np.zeros(len(STEArray))
  for index in range(len(STEArray)):
    timeSampleSTE[index] = TIME_FRAME * index / 3
  
  # Tính F0
  F0 = np.zeros(len(framesArray))
  timeSampleF0 = np.zeros(len(framesArray))
  for index in range(len(framesArray)):
      F0[index] = pitch.getPitch(index, frequency, framesArray, frameLength, markPoint)
      timeSampleF0[index] = TIME_FRAME * index / 3  
  
  F0mean = np.mean([value for value in F0 if value > 0 and value < 450])
  F0std = np.std([value for value in F0 if value > 0 and value < 450])
  
  # Show
  plt.figure(i + 1)
  plt.subplot(2, 1, 1)
  plt.title(f"Signal: {FILE_WAV_THHL[i]}")
  plt.plot(timeSample, signal)
  plt.plot(timeSampleSTE, STEArray, 'r')
  plt.axhline(y=0.04772, color='orange', linestyle='-')
  for line in LINE[i]:
    plt.axvline(line, color = 'g')
  for index in range(1, len(markPoint)):
    if markPoint[index] == 1 and markPoint[index - 1] == 0 or markPoint[index] == 0 and markPoint[index - 1] == 1:
      plt.axvline(x = TIME_FRAME * index / 3, color = 'b', linestyle = 'dashed')
  plt.legend(['Signal', 'STE', 'Threshold'])
  plt.xlabel('Time(s)')
  plt.ylabel('Signal Amplitude')
  plt.xlim([0, len(signal) / frequency])
  
  # plt.subplot(4, 1, 2)
  # plt.title(f"F0 FFT - F0mean = {round(F0mean, 2)}, F0std = {round(F0std, 2)}")
  # plt.ylim([0, 450])
  # plt.plot(timeSampleF0, F0, '.')
  # plt.xlabel('Time(s)')
  # plt.ylabel('Frequence(Hz)')
  
  # plt.subplot(4, 1, 3)
  # plt.title(f"Narrowband Of Signal: {FILE_WAV_THHL[i]}")
  # plt.specgram(signal, NFFT=1323, Fs=frequency, window=np.hamming(frameLength), noverlap=frameLength//3, cmap='jet')
  # plt.xlabel('Time(s)')
  # plt.ylabel('Frequence(Hz)')
  
  plt.subplot(2, 1, 2)
  plt.title(f"Wideband Of Signal: {FILE_WAV_THHL[i]}")
  plt.specgram(signal, NFFT=220, Fs=frequency, window=np.hamming(0.005 * frequency), noverlap=int(0.002 * frequency), cmap='jet')
  plt.xlabel('Time(s)')
  plt.ylabel('Frequence(Hz)')
  plt.tight_layout() 
plt.show()