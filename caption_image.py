# --------------------------------------------------
# load the file containing all of the captions into a single long string
# --------------------------------------------------
caption_file = "./dataset/Flickr8k_text/Flickr8k.token.txt"
image_dir = "C:/Users/nilar/Documents/emotion_based_qa/dataset/Flickr8k_Dataset/Flicker8k_Dataset/"


def load_captions(filename):
    with open(filename, "r") as fp:
        # Read all text in the file
        text = fp.read()
        return (text)


# --------------------------------------------------
# Each photo has a unique identifier, which is the file name of the image .jpg file
# Create a dictionary of photo identifiers (without the .jpg) to captions. Each photo identifier maps to
# a list of one or more textual descriptions.
#
# {"image_name_1" : ["caption 1", "caption 2", "caption 3"],
#  "image_name_2" : ["caption 4", "caption 5"]}
# --------------------------------------------------
def captions_dict(text):
    dict = {}

    # Make a List of each line in the file
    lines = text.split('\n')
    for line in lines:

        # Split into the <image_data> and <caption>
        line_split = line.split('\t')
        if (len(line_split) != 2):
            # Added this check because dataset contains some blank lines
            continue
        else:
            image_data, caption = line_split

        # Split into <image_file> and <caption_idx>
        image_file, caption_idx = image_data.split('#')
        # Split the <image_file> into <image_name>.jpg
        image_name = image_file.split('.')[0]

        # If this is the first caption for this image, create a new list for that
        # image and add the caption to it. Otherwise append the caption to the
        # existing list
        if (int(caption_idx) == 0):
            dict[image_name] = [caption]
        else:
            dict[image_name].append(caption)

    return (dict)


doc = load_captions(caption_file)
image_dict = captions_dict(doc)

# print(image_dict)

print("PHASE 1 COMPLETE")

# --------------------------------------------------
# We have three separate files which contain the names for the subset of
# images to be used for training, validation or testing respectively
#
# Given a file, we return a set of image names (without .jpg extension) in that file
# --------------------------------------------------
def subset_image_name(filename):
    data = []

    with open(filename, "r") as fp:
        # Read all text in the file
        text = fp.read()

        # Make a List of each line in the file
        lines = text.split('\n')
        for line in lines:
            # skip empty lines
            if (len(line) < 1):
                continue

            # Each line is the <image_file>
            # Split the <image_file> into <image_name>.jpg
            image_name = line.split('.')[0]

            # Add the <image_name> to the list
            data.append(image_name)

        return (set(data))


training_image_names = subset_image_name(caption_file)
# print(training_image_names)

print("PHASE 2 COMPLETE")


# -----------------------------------------------------------------
# | PHASE 3
# -----------------------------------------------------------------

import numpy as np
import tensorflow as tf


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_modelpython .layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

from tqdm import tqdm

training_image_paths = [image_dir + name + '.jpg' for name in training_image_names]

# Get unique images
encode_train = sorted(set(training_image_paths))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, batch_features.shape[3]))

    try:
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            # print("saving at", path_of_feature)
            np.save(path_of_feature, bf.numpy())
    except:
        print("error in saving feature")
        break

print("PHASE 3 COMPLETE")





