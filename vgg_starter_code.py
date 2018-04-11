import tensorflow as tf
import scipy.io
import numpy as np
import skimage
import skimage.io
import skimage.transform
import json


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

# TODO create a tensor which will contain the image to be predicted shape = [1,224,224,3]
image = None

for i in range(len(layers)):
    layer_name = layers[i][0][0][0][0]
    layer_type = layers[i][0][0][1][0]
    if layer_type == "conv":
        layer_weights = layers[i][0][0][2][0][0]
        layer_bias = layers[i][0][0][2][0][1].reshape(-1)

        # TODO create the corresponding weight variable
        w= None
        # TODO create the corresponding bias variable
        b = None

        if layer_name.startswith("conv"):
            if i == 0:

                #TODO FIRST CONVOLUTION PADDING TO BE DETERMINED
                h = None

            else:

                #TODO OTHER CONVOLUTIONS PADDING TO BE DETERMINED
                h=None

        elif layer_name.startswith('fc'):
            h = tf.map_fn(lambda w_: tf.reduce_sum(w_ * h), tf.transpose(w, perm=(3, 0, 1, 2)))
            h = h + b

    elif layer_type == "pool":
        # TODO MAXPOOLING
        h=None
    elif layer_type == "relu":
        #TODO RELU
        h = None

    elif layer_type == "softmax":
        #TODO SOFTMAX
        h = None
    else:
        raise Exception


# TODO load an image
img = load_image("image")

VGG_MEAN = [103.939, 116.779, 123.68]
img[:, :, 0] -= VGG_MEAN[0]
img[:, :, 1] -= VGG_MEAN[1]
img[:, :, 2] -= VGG_MEAN[2]

img = img[:, :, ::-1]
img = np.expand_dims(img, 0)

with open("correspondences.json") as fp:
    correspondences = json.load(fp)

correspondences = {int(k): v for k, v in correspondences.items()}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Todo get the inference
    inference = None

    inference = np.squeeze(inference)
    top_5 = inference.argsort()[-5:][::-1]

    for top in top_5:
        print(correspondences[top], inference[top])
