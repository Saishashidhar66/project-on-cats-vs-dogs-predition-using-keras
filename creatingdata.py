import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

DataDir= r"D:\shashi\py\kagglecatsanddogs_3367a\PetImages/"
CATEGORIES=["Dog","Cat"]
training_data=[]
def create_training_data():
    for i in CATEGORIES:

        path=os.path.join(DataDir,i)
        class_num=CATEGORIES.index(i)

        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(100,100))
                training_data.append([new_array,class_num])
            
            except Exception as e:
                pass


create_training_data()
print(len(training_data))
            
import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample)




features=[]
label=[]

for features,label in training_data:
    features.append(features)
    label.append(label)

print(features[0].reshape(-1,100,100,1))


features=np.array(features).reshape(-1,100,100,1)

import pickle

pickle_out=open("D:\shashi\py\Features.pickle","wb")
pickle.dump(features,pickle_out)
pickle_out.close()

pickle_out=open("D:\shashi\py\Labels.pickle","wb")
pickle.dump(label,pickle_out)
pickle_out.close()
