import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import adam
from tensorflow.keras.losses import BinaryCrossentropy

(train_dataset, test_dataset), info = tfds.load(
    'cats_vs_dogs', 
    split = ('train[:80%]', 'train[80%:]'), # this dataset only has train split
    with_info = True,
    as_supervised = True
)

def img_normalize(image, label):
    return (tf.cast(image, tf.float32)/ 255.0, label)

def resize(image, label):
    return (tf.image.resize(image, (224, 224)), label)

train_dataset = train_dataset.map(resize, num_parallel_calls = tf.data.AUTOTUNE)
train_dataset = train_dataset.map(img_normalize, num_parallel_calls = tf.data.AUTOTUNE)

shuffle_val = len(train_dataset)//1000
batch_size = 4

train_dataset = train_dataset.shuffle(shuffle_val)
train_dataset = train_dataset.batch(batch_size)

test_dataset = test_dataset.map(resize, num_parallel_calls = tf.data.AUTOTUNE)
test_dataset = test_dataset.map(img_normalize, num_parallel_calls = tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size)

def AlexNet():
    input = layers.Input((224, 224, 3))
    x = layers.Conv2D(96, 11, 4, activation= 'relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, 2)(x)
    x = layers.Conv2D(256, 5, 1, activation= 'relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, 2)(x)
    x = layers.Conv2D(384, 3, 1, activation= 'relu')(x)
    x = layers.Conv2D(384, 3, 1, activation= 'relu')(x)
    x = layers.Conv2D(256, 3, 1, activation= 'relu')(x)
    x = layers.MaxPooling2D(3, 2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(4096, activation= 'relu')
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation= 'relu')
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation= 'sigmoid')(x)
    
    model = Model(inputs= input, output= x)
    
    return model

model = AlexNet()

model.compile(
    loss = BinaryCrossentropy(),
    optimizer = adam(learning_rate= 0.001),
    metrics = ['accuracy']
)

model.fit(train_dataset, epochs= 10, validation_data = test_dataset)