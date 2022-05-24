import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from pathlib import Path

modelFullPath = './models/ukGardenModel.pb'
labelsFullPath = './models/ukGardenModel_labels.txt'

original_stdout = sys.stdout

def create_graph():
    with tf.io.gfile.GFile(modelFullPath, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(imagePath):
    answer = None

    if not tf.io.gfile.exists(imagePath):
        tf._logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.io.gfile.GFile(imagePath, 'rb').read()


    with tf.compat.v1.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print ((" {0}: {1}").format(human_string, score))
            printGraph(score)
        answer = labels[top_k[0]]
        return answer

def findImages():
    for image in os.listdir("./testImages"):
        if image.endswith(".jpg"):
            print (("Classifying {0}:").format(image))
            run_inference_on_image("./testImages/{0}".format(image))

'''def printGraph(amount):
        value = amount * 20
        sys.stdout.write(" ")
        for x in range(int(value)):
            sys.stdout.write("#")
        sys.stdout.flush()
        print ("")
'''

def printGraph(amount):
    with open ("output.txt", 'a') as f:
        value = int(amount * 20)
        #print ("", "#"*value, file=f)
        print ("")
        #f.write(''+('#*value'))
        f.write('word'+' +str(value)')

if __name__ == '__main__':
    create_graph()
    print ("graph Created")
    findImages()