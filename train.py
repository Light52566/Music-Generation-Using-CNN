import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Activation, Input
from tensorflow.keras import Model

import pandas as pd

print("TensorFlow version:", tf.__version__)

#DATASET_PATH = "final_dir/final.csv"
DATASET_PATH = "outputs/output1682.csv"

#importing the dataset
print("Importing dataset...")
df = pd.read_csv(DATASET_PATH).iloc[:,1:].replace(2, 1)
print("Splitting dataset to train and test...")
test_ds = tf.cast(tf.convert_to_tensor(df.iloc[:,df.shape[1]*6//10:]), tf.float32)
train_ds = tf.cast(tf.convert_to_tensor(df.iloc[:,:df.shape[1]*6//10]), tf.float32)
print(df.shape)



print("Defining the model...")
filter_counts = [
    [4,  8,  12, 16, 18], # block one's filter counts
    [20, 24, 28, 32], # block two's filter counts
    [34, 36, 38, 40], # block three's filter counts
]

inputs = Input(shape=(None, 1)) # None allows variable input lengths
conv = inputs

for block_idx in range(0, len(filter_counts)):
    block_filter_counts = filter_counts[block_idx]

    for i in range(0, len(block_filter_counts)):
        filter_count = block_filter_counts[i]
        dilation_rate = 2**i # exponentially growing receptive field
        conv = Conv1D(filters=filter_count, kernel_size=2,
                      strides=1, dilation_rate=dilation_rate,
                      padding='valid')(conv)
        conv = Activation('elu')(conv)

output_bias_init = tf.keras.initializers.Constant(-3.2)
outputs = Conv1D(filters=1, kernel_size=1, strides=1,
                 dilation_rate=1, padding='valid',
                 bias_initializer=output_bias_init)(conv)
tf.cast(outputs,tf.float32)
outputs = Activation('sigmoid')(outputs)

model = Model(inputs=inputs, outputs=outputs)

loss_object = tf.keras.losses.CategoricalCrossentropy()

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        
        predictions = model(images, training=True)
        #print("here")
        # print("pred:")
        # tf.print(predictions)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

print("Training model...")
EPOCHS = 5
BATCH_SIZE = 62

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    #print("train ds shape:",train_ds.shape)
    #print("no of loops",train_ds.shape[1]-BATCH_SIZE)
    for i in range(train_ds.shape[1]-BATCH_SIZE-1):
        train_batch = train_ds[:, i:i+BATCH_SIZE]
        #print(tf.shape(train_batch))
        train_label = tf.reshape(train_ds[:, i+BATCH_SIZE], [88,1,1])
        train_step(train_batch, train_label)

    #print("test ds shape:",tf.shape(test_ds))
    #print("no of loops",test_ds.shape[1]-BATCH_SIZE-1)
    for i in range(test_ds.shape[1]-BATCH_SIZE-1):
        test_batch = test_ds[:, i:i+BATCH_SIZE]
        #print("test batch shape:",tf.shape(test_batch))
        test_label = tf.reshape(test_ds[:, i+BATCH_SIZE], [88,1,1])
        test_step(test_batch, test_label)

    print(
    f'Epoch {epoch + 1}, '
    f'Loss: {train_loss.result()}, '
    f'Accuracy: {train_accuracy.result() * 100}, '
    f'Test Loss: {test_loss.result()}, '
    f'Test Accuracy: {test_accuracy.result() * 100}'
    )

#music generation
NO_OF_NOTES = 10
seed = test_ds[:, :BATCH_SIZE]
song_out = seed
for i in range(NO_OF_NOTES):
    prediction = model.predict(seed)
    song_out =tf.concat([song_out, prediction[:,:,0]])
    seed = song_out[i+1:i+BATCH_SIZE+1]
print(song_out)


