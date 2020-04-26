import cv2 
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

CATEGORIES=['Dogg','Cat']
image=r'D:\shashi\py\3.jpg'
def prepare(image):
    100=100
    img_array=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(100,100))
    return new_array.reshape(-1,100,100,1)

model = tf.keras.models.load_model(r"D:\shashi\py\Dogs_vs_Cats_CNN.model")
prediction=model.predict([prepare(image)])
print(CATEGORIES[int(prediction[0][0])])
img=mpimg.imread(image)
imgplot=plt.imshow(img)
plt.title(CATEGORIES[int(prediction[0][0])])
plt.show()
