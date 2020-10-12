import pickle

what = pickle.load(open('./fold/inD_short-train.pkl','rb'))
        #读取.pkl文件内容放入what中

import numpy, scipy.io
scipy.io.savemat('./fold/inD_short-train.mat', mdict={'what':what})