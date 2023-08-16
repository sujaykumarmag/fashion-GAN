import os

import tensorflow as tf
import matplotlib.pyplot as plt


# model.generator.load_weights(os.path.join('models', 'generatormodel.h5'))



# Load the generator model
generator_model = tf.keras.models.load_model('models/generator.h5')

# Generate images from random noise
random_noise = tf.random.normal((16, 128, 1))
imgs = generator_model.predict(random_noise)


fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10, 10))
for r in range(4): 
    for c in range(4): 
        ax[r][c].imshow(imgs[r * 4 + c])  # Correct indexing of 'imgs'

plt.savefig("results/result_gen.jpg")  # Save the figure as an image
