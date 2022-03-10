import librosa
from glob import glob
from sklearn import model_selection, svm, metrics
import numpy as np
import random

all_mfccs = []
answers = []

#keyはフォルダ名、valueはフォルダの声の主
file_and_voice = {'安倍晋三':'安倍晋三','文在寅':'文在寅','トランプ':'トランプ','プーチン':'プーチン','習近平':'習近平'}
#mfccを説明変数に、声の主を目的変数のリストに追加
for i in range(len(file_and_voice)):
    files = glob(list(file_and_voice.keys())[i]+'/*');
    voice_answer = [list(file_and_voice.values())[i]]*len(files)
    answers.extend(voice_answer)
    for file_name in files:
        x, fs = librosa.load(file_name, sr=44100)
        mfccs = librosa.feature.mfcc(x, sr=fs)
        all_mfccs.append(mfccs)
        
#説明変数と目的変数の対応関係を崩さずに順番をシャッフル
p = list(zip(all_mfccs, answers))
random.shuffle(p)
all_mfccs, answers = zip(*p)
#説明変数のリストを3次元から2次元にする
all_mfccs = np.array(all_mfccs)
all_mfccs = all_mfccs.reshape(len(all_mfccs), -1).astype(np.float64)    
#学習用とテスト用に分割する
train_size = 1100
test_size = 154
data_train, data_test, label_train, label_test = model_selection.train_test_split(all_mfccs, answers, test_size=test_size, train_size=train_size)
#svmで学習
clf = svm.SVC()
clf.fit(data_train, label_train)
#テスト用データで試す
pre = clf.predict(data_test)
ac_score = metrics.accuracy_score(label_test, pre)
#正解率を表示
print(ac_score)
    
