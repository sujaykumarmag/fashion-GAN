import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback

import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

import model_gan as model



gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu,True)

print(gpus)


# Import Dataset
df = tfds.load('fashion_mnist',split='train')

x_sample = df.as_numpy_iterator().next()

plt.imshow(x_sample.get('image'))
plt.show()
plt.savefig("visuals/sample.jpg")


data_iterator = df.as_numpy_iterator()


# To Scale the image and to take out the label
def scale_image(data):
  image = data['image']
  return image/255 # scaling [0,1]

# Data Pipeline
# map -> cache -> suffle -> batch -> prefetch
df = tfds.load('fashion_mnist',split='train')
df = df.map(scale_image)
df = df.cache()
df = df.shuffle(6000)
df = df.batch(128)
df = df.prefetch(64)

g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()



generator = model.build_generator()
img=generator.predict(np.random.randn(4,128,1))
plt.imshow(np.squeeze(img)[0])
plt.savefig("visuals/generated_sample.jpg")


discriminator = model.build_discriminator()
print(discriminator.predict(np.expand_dims(img[0],0)))
discriminator.predict(img)





# Create instance of subclassed model
fashgan = model.FashionGAN(generator, discriminator)
# Compile the model
fashgan.compile(g_opt, d_opt, g_loss, d_loss)


hist = fashgan.fit(df, epochs=200, callbacks=[model.ModelMonitor()])


plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.show()
plt.savefig("results/loss.jpg")


generator.save('models/generator.h5')
discriminator.save('models/discriminator.h5')