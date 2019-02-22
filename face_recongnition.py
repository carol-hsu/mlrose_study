# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
#import face_recognition
import argparse
import pickle
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import mlrose
from sklearn.metrics import accuracy_score
#from imutils import paths
#from neural_network import network_manager

DETECTION_METHOD="cnn"
THE_ONE="natalie_portman"

def encode_image(image_file):
    # load the input image and convert it from BGR to RGB
    image = cv2.imread(image_file)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes)

    return encodings


def load_encodings(encoding_file):
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    f = open(encoding_file, "rb")
    
    data = {}
    
    while True:
        try:
            if not data:
                data = pickle.load(f, encoding='latin1')
            else: 
                temp_data = pickle.load(f, encoding='latin1')
                data["encodings"] = data["encodings"] + temp_data["encodings"]
                data["names"] = data["names"] + temp_data["names"]
    
        except EOFError as e:
            #bad... but workable
            print("[INFO] Finish loading! Get "+str(len(data["names"]))+" faces ")
            break

    return data
#
#def data_preprocessing(dataset):
#    return 

def accuracy(predicts, answers):
    correct = len(answers)
    for i in range(len(answers)):
        if (predicts[i] == "unknown") or (predicts[i] != answers[i]):
            correct-=1
    print(correct/float(len(answers))*100)

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--train", required=True,
            help="directory of training set/encodings of training set")
    ap.add_argument("-t", "--test", required=True,
            help="directory of testing set/encodings of testing set")
    params = vars(ap.parse_args())
    
    
    faces = load_encodings(params["train"])
    test_faces = load_encodings(params["test"])
    # preprocess data
#    X_train, X_test, y_train, y_test = train_test_split(faces["encodings"], faces["names"], \
#                                                        test_size=0.2, random_state = 1)


    # make target to binary list
    X_train = faces["encodings"]
    X_test = test_faces["encodings"]
    y_train = [ 1 if y == THE_ONE else 0 for y in faces["names"] ]
    y_test = [ 1 if y == THE_ONE else 0 for y in test_faces["names"] ]

    np.random.seed(1)
    # build neural network
    nn_model = mlrose.NeuralNetwork(hidden_nodes = [50, 20, 8, 2], activation = activ[param["activation"]], \
                                 algorithm = algos[param["algo"]], max_iters = 1000, \
                                 #algorithm = 'gradient_descent', max_iters = 1000, \
                                 bias = True, is_classifier = True, learning_rate = 0.001, \
                                 early_stopping = True, clip_max = 10, max_attempts = 100)

#    for i in range(6):
#        print(i)
#        X_train, X_tmp, y_train, y_tmp = train_test_split(X_init_train, y_init_train, \
#                                                            test_size=0.1, random_state = i)
#        if len(nn_model.fitted_weights):
#            train_pred = nn_model.predict(X_train)
#            print(accuracy_score(y_train, train_pred))
#            #fit different training set each time
#            nn_model.fit(X_train, y_train, nn_model.fitted_weights)
#        else:
#            nn_model.fit(X_train, y_train)
    
    nn_model.fit(X_train, y_train)

    test_pred = nn_model.predict(X_test)
    print(accuracy_score(y_test, test_pred))
