import tensorflow as tf
import scipy.io
import numpy as np
import skimage
import skimage.io
import skimage.transform


def load_image(path):
    img = skimage.io.imread(path)
    img = img / 1.0
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


vgg_path = "./imagenet-vgg-verydeep-19.mat"
model = scipy.io.loadmat(vgg_path)
layers = model['layers'][0]

image = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="image")
for i in range(len(layers)):
    layer_name = layers[i][0][0][0][0]
    layer_type = layers[i][0][0][1][0]
    if layer_type == "conv":
        layer_weights = layers[i][0][0][2][0][0]
        layer_bias = layers[i][0][0][2][0][1].reshape(-1)
        w = tf.get_variable(layer_name + '/w', initializer=tf.constant(layer_weights), trainable=False)
        b = tf.get_variable(layer_name + '/b', initializer=tf.constant(layer_bias), trainable=False)
        if layer_name.startswith("conv"):
            if i == 0:
                h = tf.nn.conv2d(image, w, strides=[1, 1, 1, 1], padding='SAME') + b
            else:
                h = tf.nn.conv2d(h, w, strides=[1, 1, 1, 1], padding="SAME") + b
        elif layer_name.startswith('fc'):
            h = tf.map_fn(lambda w_: tf.reduce_sum(w_ * h), tf.transpose(w, perm=(3, 0, 1, 2)))
            h = h + b

    elif layer_type == "pool":
        h = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    elif layer_type == "relu":
        h = tf.nn.relu(h)

    elif layer_type == "softmax":
        h = tf.nn.softmax(h)
    else:
        raise Exception

    if layer_name == "pool5":  # Here we stop at pool 5 but you can do whatever you want
        break

_, height, width, depth = h.get_shape().as_list()

h = tf.reshape(h, shape=(-1, height * width * depth))

# TODO do your magic

X_train = np.load("./X_train.npy")
Y_train = np.load("./Y_train.npy")
n_train = len(X_train)

X_val = np.load("./X_val.npy")
Y_val = np.load("./Y_val.npy")
n_val = len(X_val)

X_test = np.load("./X_test.npy")
Y_test = np.load("./Y_test.npy")
n_test = len(X_test)


# TODO do some preprocessing either in numpy or for experts in tensorflow
VGG_MEAN = [103.939, 116.779, 123.68]

batch_size = None
epochs = None
index_train = np.arange(n_train)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for ep in range(epochs):
        # Training :
        np.random.shuffle(index_train)
        X_train = X_train[index_train]
        Y_train = Y_train[index_train]
        for batch_start, batch_end in zip(range(0, n_train - batch_size, batch_size),
                                          range(batch_size, n_train, batch_size)):
            _, l_train, precision_train = sess.run([optimizer, loss, precision], feed_dict={})

    # Validation :
        for batch_start, batch_end in zip(range(0, n_train - batch_size, batch_size),
                                          range(batch_size, n_train, batch_size)):
            _, l_val, precision_val = sess.run([loss, precision], feed_dict={})



    # Testing and asses the quality of your model
    for batch_start, batch_end in zip(range(0, n_test - batch_size, batch_size),
                                          range(batch_size, n_test, batch_size)):
            precision_test = sess.run(precision, feed_dict={})



