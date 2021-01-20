"""
YOLO Car Detection
@author: Bari Arviv
"""
import cv2
import json
import classifier
import numpy as np


def init_label_color_lists():
    labels_path = "yolo-coco/coco.names"
    
    # load the COCO class labels and init a list of color
    np.random.seed(42)
    label_list = open(labels_path).read().strip().split("\n") 
    color_list = np.random.randint(0, 255, size=(len(label_list), 3), dtype="uint8")
    
    return label_list, color_list


def init_config():
    weights_path = "yolo-coco/yolov3.weights"
    config_path = "yolo-coco/yolov3.cfg"

    # load YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    
    # determine only the output layer names from YOLO
    layer_names = net.getLayerNames()
    output_layers = [layer_names[idx[0] - 1] for idx in net.getUnconnectedOutLayers()]
    return net, output_layers


def init_output_layer(image, net, output_layers):
    # construct a blob, give the bounding boxes and associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    return net.forward(output_layers)


def detection(outputs, threshold_non_maxima, min_prob, in_h, in_w):
    id_list = []
    boxes = []
    probabilities = []
    non_maxima = []
    
    for output in outputs:
        for detection in output:
            # extract the class ID and probability of the object detection
            scores = detection[5:]
            id = np.argmax(scores)
            confidence = scores[id]
    
            # filter out weak predictions 
            if confidence > min_prob:
                # scale the bounding box coordinates 
                box = detection[0:4] * np.array([in_w, in_h, in_w, in_h])
                (middle_x, middle_y, width, height) = box.astype("int")
    
                # use the center to derive the top and left corner of the bounding box
                x = int(middle_x - (width / 2))
                y = int(middle_y - (height / 2))
    
                boxes.append([x, y, int(width), int(height)])
                probabilities.append(float(confidence))
                id_list.append(id)
                
    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    non_maxima = cv2.dnn.NMSBoxes(boxes, probabilities, min_prob, threshold_non_maxima)
    return boxes, probabilities, id_list, non_maxima


def predict_color(image, non_maxima_list, boxes, labels, colors, id_list, probabilities):
    # ensure at least one detection exists
    if len(non_maxima_list) == 0:
        return
    
    jsdata = {}
    car_color = classifier.Classifier()
    font = cv2.FONT_HERSHEY_SIMPLEX
        
    for i in non_maxima_list.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[id_list[i]]]
        
        if id_list[i] == 2:
            result = car_color.predict(image[max(y, 0):y + h, max(x, 0):x + w],
                                       jsdata, [x, y, x + w, y + h])
            text = "{}: {:.4f}".format(result[0]['make'], float(result[0]['prob']))
            cv2.putText(image, text, (x + 2, y + 20), font, 0.5, color, 2)
            cv2.putText(image, result[0]['model'], (x + 2, y + 40), font, 0.5, color, 2)
                
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(labels[id_list[i]] + str(len(jsdata)), probabilities[i])
        cv2.putText(image, text, (x, y - 5), font, 0.6, color, 2)
        
    return image, jsdata


def over_frames(threshold_non_maxima, min_prob, labels, colors, net, output_layers):
    # init the video stream, pointer to output video file, and frame dimensions
    vs = cv2.VideoCapture('inputs/input_video.mp4')
    writer = None
    (width, height) = (None, None)
    
    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        
        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break
    
        # if the frame dimensions are empty, grab them
        if width is None or height is None:
            (height, width) = frame.shape[:2]
        
        outputs = init_output_layer(frame, net, output_layers)
        boxes, probabilities, id_list, non_maxima_list = detection(outputs, threshold_non_maxima,
                                                                   min_prob, height, width)
        frame, jsdata = predict_color(frame, non_maxima_list, boxes, labels, colors, id_list, probabilities)
        
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('output_video.avi', fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)
    
    if writer:
        writer.release()
    vs.release()


def save_json(file_name, data):
    # saving the results in a JSON file
    with open(file_name + '.json', 'w+') as fp:
        json.dump(data, fp)


def show_output_image(image, width, height):
    # show the output image
    cv2.namedWindow('YOLO Output Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLO Output Image', width, height)
    cv2.imshow("YOLO Output Image", image)
    cv2.imwrite("output_image.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def main():
    min_prob = 0.7
    threshold_non_maxima = 0.3
    labels, colors = init_label_color_lists()
    
    ## q1
    input_image = cv2.imread('inputs/input_image.jpg')
    (input_height, input_width) = input_image.shape[:2]
    net, output_layers = init_config()
    outputs = init_output_layer(input_image, net, output_layers)
    
    boxes, probabilities, id_list, non_maxima_list = detection(outputs, threshold_non_maxima,
                                                               min_prob, input_height, input_width)
    input_image, jsdata = predict_color(input_image, non_maxima_list, boxes, labels,
                                        colors, id_list, probabilities)
    show_output_image(input_image, input_width, input_height)
    
    ## q2
    save_json('data_output_image', jsdata)
    
    ## q3
    over_frames(threshold_non_maxima, min_prob, labels, colors, net, output_layers)
    
if __name__ == "__main__":
    main()