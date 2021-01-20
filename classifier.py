import cv2
import config
import numpy as np
import tensorflow.compat.v1 as tf

model_file = config.model_file
label_file = config.label_file
input_layer = config.input_layer
output_layer = config.output_layer
input_size = config.classifier_input_size

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
    
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def load_labels(label_file):
    labels = []
    with open(label_file, "r", encoding='cp1251') as inst:
        for line in inst:
            labels.append(line.rstrip())
    return labels

def resize_and_pad(image, size, padColor=0):
    height, width = image.shape[:2]
    size_height, size_width = size
    pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    
    # interpolation method
    if height > size_height or width > size_width:
        interpolation = cv2.INTER_AREA
    else: # stretching image
        interpolation = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = width / height 

    # compute scaling and pad sizing
    if aspect > 1: # horizontal
        new_width = size_width
        new_height = np.round(new_width / aspect).astype(int)
        pad_vert = (size_height - new_height) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
    elif aspect < 1: # vertical
        new_height = size_height
        new_width = np.round(new_height * aspect).astype(int)
        pad_horz = (size_width - new_width) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
    else: # square
        new_height, new_width = size_height, size_width
        
    # scale and pad
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    scaled_image = cv2.copyMakeBorder(scaled_image, pad_top, pad_bot, pad_left, pad_right, 
                                      borderType=cv2.BORDER_CONSTANT, value=padColor)
    return scaled_image

class Classifier():
    def __init__(self):
        self.graph = load_graph(model_file)
        self.labels = load_labels(label_file)
        self.input_operation = self.graph.get_operation_by_name("import/" + input_layer)
        self.output_operation = self.graph.get_operation_by_name("import/" + output_layer)
        self.sess = tf.Session(graph=self.graph)
        self.sess.graph.finalize()

    def predict(self, image, jsdata, coordinates):
        jsdata['car' + str(len(jsdata) + 1)] = coordinates
        
        image = image[:, :, ::-1]
        image = resize_and_pad(image, input_size)
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image /= 127.5
        image -= 1.

        results = self.sess.run(self.output_operation.outputs[0], 
                                {self.input_operation.outputs[0]: image})
        results = np.squeeze(results)

        top = 3
        top_indices = results.argsort()[-top:][::-1]
        classes = []
        
        for ix in top_indices:
            make_model = self.labels[ix].split('\t')
            classes.append({"make": make_model[0], "model": make_model[1],
                            "prob": str(results[ix])})
        return(classes)