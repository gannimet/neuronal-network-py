from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import random

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Shape Trainingsdaten: {}".format(train_images.shape))
print("Dimension Bild Nr. 5: {}".format(train_images[5].shape))
print("Label zu Bild Nr. 5: {}".format(train_labels[5]))

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')
train_images /= 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')
test_images /= 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

nr_train_images = train_images.shape[0]
nr_test_images = test_images.shape[0]

print("Trainingsdatensatz: {}".format(train_images.shape))
print("Testdatensatz: {}".format(test_images.shape))
print("Wir haben {} Trainingsbilder und {} Testbilder.".format(nr_train_images, nr_test_images))

# Zufallszahl zwischen 0 und 60000
# da unser Trainingsdatensatz 60000 Bilder umfasst
randindex = random.randint(0, 60000)
plt_title = "Trainingsbild Nr. {}\nKlasse: {}".format(randindex, train_labels[randindex])
plt.imshow(train_images[randindex].reshape(28, 28), cmap='gray')
plt.title(plt_title)
plt.axis('off')
plt.show()
