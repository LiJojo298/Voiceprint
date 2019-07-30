'''
对输入的语音数据序列（PCM 码流）进行预处理。
a) 去除非语音信号 和 静默语音信号；
b) 对语音信号分帧，以供后续处理。
提取每一帧语音信号的MFCC 参数 并保存。
1)预增强（Pre-Emphasis） ：差分语音信号。
2)音框化（Framing） ：对语音数据分帧。
3)汉明窗（Hamming Windowing） ：对每帧信号加窗，以减小吉布斯效应的影响。
4)快速傅立叶变换（FFT） ：将时域信号变换成为信号的功率谱。
5)三角带通滤波器（Triangle Filters） ：三角滤波器覆盖的范围都近似于人耳的一个临界带宽，以此来 模拟人耳的掩蔽效应。
6)离散余弦转换（DCT） ：去除各维信号之间的相关性，将信号映射到低维空间。
用第2步提取的 MFCC 参数训练话者的 GMM （高斯混合模型），得到专属某话者的 GMM 声纹模型。
声纹识别。提供输入话音与GMM 声纹模型的匹配运算函数，以判断输入话音是否与声纹匹配。
'''
import librosa
import numpy as np
import librosa.display
from scipy.fftpack import dct
import matplotlib.pyplot as plt


# 读取语音数据，返回信号和频率
def audio_read(path):
    sig, fre = librosa.load(path, sr=None)
    return sig, fre


# 预加重  差分实现y(t)=x(t)−αx(t−1)
def audio_preemphasis(sig, alph=0.97):
    sig_new = np.append(sig[0], sig[1:] - alph * sig[:-1])
    return sig_new


# 分帧
def audio_frame(sig, sample_rate, frame_size=0.025, frame_stride=0.01):

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(sig)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length

    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(sig, z) # 补零

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T  #

    frames = pad_signal[np.mat(indices).astype(np.int32, copy=False)]  # 矩阵索引

    print(pad_signal[:10])

    print(frames[2,:10])
    return frames

# 加窗 # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))
def audio_wind(frames,wind=np.hamming):
    return frames[:,]*wind(frames.shape[1])

# fft
def audio_fft(frames, NFFT = 512):
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    return pow_frames


# 滤波器组
def audio_filter(frames,sample_rate,nfilt =40, NFFT=512):
    low_freq_mel = 0
    # 将频率转换为Mel
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])            # center
        f_m_plus = int(bin[m + 1])   # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    filter_banks = np.dot(audio_fft(frames, NFFT), fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability: log0 error
    filter_banks = 20 * np.log10(filter_banks)  # dB
    # filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    return filter_banks

# mfcc
def audio_mfcc(filter_banks):
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape

    # 正弦升降1应用于MFCC以降低已被声称在噪声信号中改善语音识别的较高MFCC
    n = np.arange(ncoeff)
    cep_lifter = 22
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  # *

    # filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)  # 平均归一化MFCC
    return mfcc

if __name__ == "__main__":
    sig, fre = audio_read(librosa.util.example_audio_file())
    sig_new = audio_preemphasis(sig)
    frames = audio_frame(sig_new, fre)

    mfcc = librosa.feature.mfcc(sig,fre,n_mfcc=23)
    # frames_new = audio_wind(frames)
    # # print(frames_new.shape)
    #
    # filter_banks = audio_filter(frames_new, fre)
    # mfcc = audio_mfcc(filter_banks)
    # plt.imshow(np.flipud(mfcc.T), cmap=plt.cm.jet, aspect=0.2, extent=[0, mfcc.shape[0], 0, mfcc.shape[1]])  # 热力图
    # plt.show()


    # plt.subplot(2, 1, 1)
    # librosa.display.waveplot(sig, sr=fre)
    # plt.title('original wavform')
    #
    # plt.subplot(2, 1, 2)
    # librosa.display.waveplot(sig_new, sr=fre)
    # plt.title('preemphasis wavform')
    # plt.tight_layout()  # 保证图不重叠
    # plt.show()
