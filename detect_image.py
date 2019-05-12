from object_detection import ObjectDetector
import cv2
import sys
import logging
import argparse
import os

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"

classes = ["background", "aeroplane", "bicycle", "bird", "boat",
     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
     "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
     "sofa", "train", "tvmonitor"]


def output_filepath(filepath, output_suffix):
    filename, file_extension = os.path.splitext(filepath)
    return filename + output_suffix + file_extension

def detect_image(detector, filepath, output_filepath):
    logging.info('processing: %s', filepath)
    image = cv2.imread(filepath)
    detections = detector.detect_objects(image)
    detector.draw_detections(image, detections) 
    cv2.imwrite(output_filepath, image)  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file_paths', nargs='*', default=sys.stdin, help='filepaths to the images for object detection.')
    parser.add_argument('-l', '--logfile', help='filepath for logging output, default is stdout.')
    parser.add_argument('-o', '--outputsuffix', help='output image filename is made up of the input filepath with this suffix inserted before the file extention.', default='_output')
    parser.add_argument('-c', '--confidencethreshold', help='number between 0 and 1 that represents the confidence level for declaring an object as detected. e.g. .5 requires greater than 50%% confidence.', default=0.5, type=float) 
    args = parser.parse_args()
    
    if args.logfile is None:
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(filename=args.logfile, format='%(message)s', level=logging.DEBUG)    

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    detector = ObjectDetector(net, classes, args.confidencethreshold)

    for f in args.image_file_paths:
        filepath = f.strip()
        detect_image(detector, filepath, output_filepath(filepath, args.outputsuffix))
    


if __name__ == "__main__":
    main()
