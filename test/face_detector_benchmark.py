import numpy as np
import time
import cv2, argparse, os, sys
from tqdm import tqdm
import torch
<<<<<<< HEAD
from facenet_pytorch import MTCNN
=======
sys.path.append(os.getcwd())
>>>>>>> e0abb45334332f2d484258bbe428507ce9d37712
import face_detection
from face_detection.api import LandmarksType

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Inference code to benchmark face detector')
parser.add_argument('--batch_size', type=int, help='Batch size for face detector.', default=32, required=False)
parser.add_argument('--image', type=str, help="image to conduct face detection.", default="test/person.jpg", required=False)
parser.add_argument('--num_images', type=int, help="", default=1000, required=True)
parser.add_argument('--method', type=str, help="MTCNN or S3FD", default="mtcnn", required=False)


if __name__ == "__main__":
    args = parser.parse_args()
    if type(args.batch_size) is not int or args.batch_size < 1:
        raise ValueError("Batch size was required is a integer type.")
    if args.method not in ['mtcnn', 's3fd']:
        raise ValueError("Method invalid.")

    load_start = time.time()
    if args.method == 's3fd':
        detector = face_detection.FaceAlignment(LandmarksType._2D, flip_input=False, device=device)
    else:
        detector = MTCNN()
    load_end = time.time()

    det_start = time.time()
    for i in tqdm(range(0, args.num_images, args.batch_size), total=int(np.ceil(args.num_images/args.batch_size))):
        if not args.num_images > i + args.batch_size:
            batch_image = [cv2.imread(args.image) for k in range(i, args.num_images)]
        else:
            batch_image = [cv2.imread(args.image) for _ in range(args.batch_size)]

        if args.method == 's3fd':
            predictions = detector.get_detections_for_batch(np.asarray(batch_image))
        else:
            predictions = detector.detect(np.asarray(batch_image))

    det_end = time.time()

    print("Load model time: {} s".format(load_end-load_start))
    print("Inference total {} images in {} s".format(args.num_images, det_end-det_start))




