from imutils import paths
from cv2 import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# If LOAD is true the program will run using the model saved at ./model
# Else if LOAD is false the program with train a new model and run on that
# the new model will replace the model at ./model
LOAD = True

INIT_LR = 1e-3
EPOCHS = 25
BATCH_SIZE = 8
DIMENSION_SIZE = 224

PATH_TO_COVID_DATASET = './Training/Covid/'
PATH_TO_NORMAL_DATASET = './Training/Normal/'

COVID = 'covid'
NON_COVID = 'non-covid'

def label_images(label, imagePaths):
    data = []
    labels = [label] * len(imagePaths)

    # Scale images to be the same size and retrieve image data
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (DIMENSION_SIZE, DIMENSION_SIZE))

        data.append(image)
    return data, labels

def get_training_data():
    print("[INFO] loading training and test data...")
    covid_image_paths = list(paths.list_images(PATH_TO_COVID_DATASET))
    normal_image_paths = list(paths.list_images(PATH_TO_NORMAL_DATASET))

    data = []
    labels = []

    covid_data, covid_labels = label_images(COVID, covid_image_paths)
    normal_data, normal_labels = label_images(NON_COVID, normal_image_paths)

    data.extend(covid_data)
    data.extend(normal_data)
    labels.extend(covid_labels)
    labels.extend(normal_labels)

    data = np.array(data) / 255.0
    labels = np.array(labels)
    return data, labels

def partition_training_data(data, labels):
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    return train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=3)

def get_head_conv_block():
    return Sequential([
        layers.Input(shape=(DIMENSION_SIZE, DIMENSION_SIZE, 3)),
        layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPool2D(pool_size=(2, 2))
    ])

def get_conv_block(filters):
    return Sequential([ 
        layers.SeparableConv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.SeparableConv2D(filters=filters, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(2, 2)),
    ])

def get_FC_layer():
    return Sequential([
        layers.AveragePooling2D(pool_size=(4, 4)),
        layers.Flatten(name="flatten"),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(2, activation="softmax"),
    ])

def build_model():
    # Define each sequential layer of the model
    model = Sequential([
        get_head_conv_block(),
        get_conv_block(32),
        get_conv_block(64),
        get_conv_block(128),
        layers.Dropout(rate=0.2),
        get_conv_block(128),
        layers.Dropout(rate=0.2),
        get_FC_layer(),
        layers.Dense(units=2, activation='softmax')
    ])

    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="binary_crossentropy", optimizer=opt,
        metrics=["accuracy"])

    return model

def train_model(model, trainX, testX, trainY, testY):
    trainAug = ImageDataGenerator(
        rotation_range=15,
	    fill_mode="nearest")

    print("[INFO] training model...")
    return model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=BATCH_SIZE),
        steps_per_epoch=len(trainX) // BATCH_SIZE,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BATCH_SIZE,
        epochs=EPOCHS)

def test_model(model, testX, testY):
    # make predictions on the testing set
    print("[INFO] evaluating network...")
    predIdxs = model.predict(testX, batch_size=BATCH_SIZE)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    print(classification_report(testY.argmax(axis=1), predIdxs,
        target_names=[COVID, NON_COVID]))

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

def plot_training_results(H):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)
    plt.legend(loc="lower left")
    plt.savefig('./lots/loss.png')
    plt.figure()
    plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend(loc="lower left")
    plt.savefig('./Plots/accuracy.png')

def save_model(model):
    print("[INFO] saving COVID-19 detector model...")
    model.save('model', save_format="h5")

def main(load=False):
    data, labels = get_training_data()
    (trainX, testX, trainY, testY) = partition_training_data(data, labels)
    
    model = None
    if load:
        model = load_model('model')
    else:
        model = build_model()
    
        H = train_model(model, trainX, testX, trainY, testY)
        plot_training_results(H)

    test_model(model, testX, testY)
    plot_model(model, to_file='./Plots/model.png', show_shapes=True, show_layer_names=False)
    
    save_model(model)

main(load=LOAD)
