import numpy as np
import cv2
import logging

class ObjectDetector:

     def __init__(self, net, classes, confidence_threshold):
         self.net = net
         self.classes = classes
         self.confidence_threshold = confidence_threshold
         self.colors = np.random.uniform(0, 255, size=(len(classes), 3))

     def detect_objects(self, image):
          blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
          self.net.setInput(blob)
          return self.net.forward()

          
     def draw_detections(self, image, detections):
          (h, w) = image.shape[:2]
          for detection_index in np.arange(0, detections.shape[2]):
               confidence = detections[0, 0, detection_index, 2]

               if confidence > self.confidence_threshold:
                    class_index = int(detections[0, 0, detection_index, 1])
                    box = detections[0, 0, detection_index, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(self.classes[class_index], confidence * 100)
                    logging.info("%s  box: %d, %d, %d, %d", label, startX, startY, endX, endY )
                    cv2.rectangle(image, (startX, startY), (endX, endY), self.colors[class_index], 2)
                    labelY = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(image, label, (startX, labelY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[class_index], 2)

          return image    
