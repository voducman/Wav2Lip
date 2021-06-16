import numpy as np
import cv2
from facenet_pytorch import MTCNN

if __name__ == "__main__":
    img = cv2.cvtColor(cv2.resize(cv2.imread("test/family.jpg"), (900, 700)), cv2.COLOR_BGR2RGB)
    img_1 = cv2.cvtColor(cv2.resize(cv2.imread("test/person.jpg"), (900, 700)), cv2.COLOR_BGR2RGB)
    batch_img = [img, img, img_1]
    detector = MTCNN()
    predictions = detector.detect(batch_img)
    print(predictions, predictions[1][2].shape)
    # if len(predictions) > 0:
    #     prediction = predictions[0]['box']
    #     conf = predictions[0]['confidence']
    #     print("Bbox:", prediction, "--", "Conf:", conf)

