import argparse
import os
import sys
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from keras.models import load_model

#################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, default='',
                    help='path to image file')
parser.add_argument('--video', type=str, default='',
                    help='path to video file')
parser.add_argument('--src', type=int, default=0,
                    help='source of the camera')
args = parser.parse_args()
#################################################################

class FaceDetector(object):
    '''
    Face Detector class
    '''
    def __init__(self, mtcnn, emotion_classifier, emotions, data):
        self.mtcnn = mtcnn
        self.emotion_classifier = emotion_classifier
        self.emotions = emotions
        self.data = data
    
    def _draw(self, frame, boxes, probs):
        '''
        Draw bounding box and probs
        '''
        for box, prob in zip(boxes, probs):
            # draw rectangle on frame
            cv2.rectangle(frame, (box[0], box[1]),
                                 (box[2], box[3]),
                                 (0,0,255), thickness=2)
            # show probability
            cv2.putText(frame, str(prob), (box[2], box[3]),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1,
                        cv2.LINE_AA)
        return frame

    def _detect_rois(self, boxes):
        '''
        return rois as a list
        '''
        rois = list()
        for box in boxes:
            roi = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
            rois.append(roi)
        return rois
    
    def emotion_class(self, face):
        # convert color to gray scale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # resize the image to 64x64 for neural network
        f = cv2.resize(gray, (64,64))
        f = f.astype('float')/255.0
        f = np.array(f)
        f = np.expand_dims(f, axis=0)
        f = f[:,:,:,np.newaxis]

        # emotion predict
        preds = self.emotion_classifier.predict(f)[0]
        label = self.emotions[preds.argmax()]
        return label

    def run(self, data):
        # video capture using data
        cap = cv2.VideoCapture(data)

        while True:
            # capture image from camera
            ret, frame = cap.read()

            try:
                # detect face box and landmarks
                boxes, probs = self.mtcnn.detect(frame, landmarks=False)

                # draw on frame
                self._draw(frame, boxes, probs)

                # perform emotion recognition & gender classification only when face is detected
                if len(boxes) > 0:

                    # extract the face roi
                    rois = self._detect_rois(boxes)

                    for roi in rois:
                        (start_Y, end_Y, start_X, end_X) = roi
                        face = frame[start_Y:end_Y, start_X:end_X]

                        # run the classifier on bounding box
                        emo = self.emotion_class(face)

                        # assign labeling
                        cv2.putText(frame, emo, (start_X, start_Y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
            except:
                pass

            # show the frame
            window = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            cv2.imshow('Face detect', window)
            
            # save image
            # cv2.imwrite('sample/sample.jpg', window)

            # q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Interrupted by user!')
                break
        
        # clear program and close windows
        cap.release()
        cv2.destroyAllWindows()
        print('All done!')

mtcnn = MTCNN()

emotion_classifier = load_model('trained_model\\fer2013_mini_XCEPTION.107-0.66.hdf5')
EMOTIONS = ['Angry', 'Disgusting', 'Fearful', 'Happy', 'Sad', 'Surprise', 'Neutral']

if args.image:
    if not os.path.isfile(args.image):
        print("Input image file {} doesn't exist".format(args.image))
        sys.exit(1)
    fcd = FaceDetector(mtcnn, emotion_classifier, EMOTIONS, args.image)
    fcd.run(args.image)
elif args.video:
    if not os.path.isfile(args.video):
        print("Input video file {} dosen't exist".format(args.video))
        sys.exit(1)
    fcd = FaceDetector(mtcnn, emotion_classifier, EMOTIONS, args.video)
    fcd.run(args.video)
else:
    fcd = FaceDetector(mtcnn, emotion_classifier, EMOTIONS, args.src)
    fcd.run(args.src)
