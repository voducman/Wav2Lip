import numpy as np
import cv2
from facenet_pytorch import MTCNN

if __name__ == "__main__":
    img = cv2.cvtColor(cv2.resize(cv2.imread("test/family.jpg"), (900, 700)), cv2.COLOR_BGR2RGB)
    img_1 = cv2.cvtColor(cv2.resize(cv2.imread("test/orange.jpg"), (900, 700)), cv2.COLOR_BGR2RGB)
    # batch_img = [img, img, img_1]
    batch_img = [img_1]
    detector = MTCNN()
    predictions = detector.detect(batch_img)
    print(predictions)
    if predictions is not None and len(predictions) > 0:
        bboxes = predictions[0]
        confs = predictions[1]
        for i, boxes in enumerate(bboxes):
            image = batch_img[i]
            for box in boxes:
                (x1, y1, x2, y2) = box.astype(np.int32)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), color=(255,0,0), thickness=3)
            cv2.imwrite(f"test/{i}.jpg", image)

