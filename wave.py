import pywt
import numpy as np

def wavelet_transform(feature_vector, wavelet='db1', level=1):
    # 小波变换

    coeffs = pywt.wavedec(feature_vector, wavelet, level=level)

    # 提取低频部分（近似部分）
    low_freq = coeffs[0]

    # 提取高频部分（细节部分）
    high_freq = np.concatenate(coeffs[1:])

    return low_freq, high_freq




