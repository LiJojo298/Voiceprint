"""
基于MFCC的GMM声纹识别

"""
import os
import numpy as np
import math
import operator
import time
import pickle
from scipy.io import wavfile
from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture
from collections import defaultdict


class Model:

    def __init__(self):
        self.gmms = []
        self.labes = []

    # 读取语音
    def read_wav(self, fname):
        fs, signal = wavfile.read(fname)
        if len(signal.shape) != 1:
            print("convert stereo to mono")
            signal = signal[:, 0]
        return fs, signal

    # 获取mfcc特征
    def get_feature(self, fs, signal):
        mfcc_feature = mfcc(signal, fs)
        if len(mfcc_feature) == 0:
            print("ERROR.. failed to extract mfcc feature:%s" % len(signal))
        return mfcc_feature

    # 高斯混合模型
    def GMM(self, feats, gmm_order=32):
        gmm = GaussianMixture(gmm_order)
        gmm.fit(feats)
        return gmm

    # train
    def train(self, input_dirs, output_model_filename):
        features = defaultdict(list)
        if not os.path.exists(input_dirs):
            print("path error !")
        for root, dirs, names in os.walk(input_dirs):
            for filename in names:
                portion = os.path.splitext(filename)
                # 根据后缀来修改
                if portion[1] == ".wav":
                    label = portion[0]
                    fs, signal = self.read_wav(os.path.join(root, filename))
                    mfcc = self.get_feature(fs, signal)
                    features[label].extend(mfcc)
        start_time = time.time()
        for label, feats in features.items():
            try:
                self.labes.append(label)
                self.gmms.append(self.GMM(feats))
            except Exception as e:
                print("%s:%s failed" % (e, label))
        print(time.time() - start_time, " seconds")
        self.dump(output_model_filename)

    # predict
    def predict(self, input_files, input_model):
        m = self.load(input_model)

        fs, signal = self.read_wav(input_files)
        mfcc = self.get_feature(fs, signal)

        scores = [self.gmm_score(gmm, mfcc) / len(mfcc) for gmm in m.gmms]
        p = sorted(enumerate(scores), key=operator.itemgetter(1), reverse=True)

        p = [(str(m.labes[i]), y, p[0][1] - y) for i, y in p]
        result = [(m.labes[index], value) for (index, value) in enumerate(scores)]
        p = max(result, key=operator.itemgetter(1))
        softmax_score = self.softmax(scores)
        label = p[0]
        score = softmax_score
        print(input_files, '->', label, ", score->", score)
        return label, score

    def gmm_score(self, gmm, x):
        return np.sum(gmm.score(x))

    def softmax(self,scores):
        scores_sum = sum([math.exp(i) for i in scores])
        score_max = math.exp(max(scores))
        return round(score_max / scores_sum, 3)

    def load(self, fname):
        with open(fname, 'rb') as f:
            R = pickle.load(f)
            return R

    def dump(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

#
if __name__ == "__main__":
      m = Model()
      wav_path = r'E:\study\文献-声纹识别文献\远场语音'
      model_path = r'E:\study\文献-声纹识别文献\远场语音\model.out'

      predict_wav_path = r'E:\study\文献-声纹识别文献\远场语音\201907191112061818032248_K.wav'
      # m.train(wav_path, model_path)

      m.predict(predict_wav_path, model_path)
