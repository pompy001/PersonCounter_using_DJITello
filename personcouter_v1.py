from djitellopy import tello
import cv2
import math
from time import sleep

import kepressmodule as kp
from typing import Counter
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np




class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox = OrderedDict()  # CHANGE

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.maxDistance = maxDistance

    def register(self, centroid, inputRect):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.bbox[self.nextObjectID] = inputRect  # CHANGE
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]  # CHANGE
        

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            # return self.objects
            return self.bbox

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        inputRects = []
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            inputRects.append(rects[i])  # CHANGE

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], inputRects[i])  # CHANGE

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputRects[col]  # CHANGE
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], inputRects[col])

        # return the set of trackable objects
        # return self.objects
        return self.bbox

class FaceDetector:
    """
    Find faces in realtime using the light weight model provided in the mediapipe
    library.
    """

    def __init__(self, minDetectionCon=0.7):
        """
        :param minDetectionCon: Minimum Detection Confidence Threshold
        """

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)
        self.count=0
        self.tracker=CentroidTracker(maxDisappeared=30,maxDistance=90)
        self.obj_id=[]

    def non_max_suppression_fast(self,boxes, overlapThresh):
    
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    

    def findFaces(self, img, draw=True):
        """
        Find faces in an image and return the bbox info
        :param img: Image to find the faces in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings.
                 Bounding Box list.
        """
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        
        box=[]
        
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, = img.shape[0],img.shape[1]
                bboxcv = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int((bboxC.width+bboxC.xmin) * iw), int((bboxC.height+bboxC.ymin) * ih)
                box.append(bboxcv)
              
            boundingbox = np.array(box)
            boundingbox = boundingbox.astype(int)
            box = self.non_max_suppression_fast(boundingbox,0.3)
            objects = self.tracker.update(box)
            # print(objects)
            for (objID,bboxes) in objects.items():
                # print(objID)
                

                x1,y1,x2,y2 = bboxes

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2=  int(y2)
                
                

                
                print(bboxes)
                img = cv2.rectangle(img, (x1,y1),(x2,y2), (255, 0, 255), 2)
                text = "ID :{}".format(objID)
                cv2.putText(img, text,(x1,y1-5), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255), 2)
                if objID not in self.obj_id:
                    self.obj_id.append(objID)
                    print(self.obj_id)
        return img,self.obj_id 



## PARAMETERS
fSpeed = 117/18 #forward speed cm/s actual(15 cm/s)
aSpeed = 360/10 # angilar speed Degree/s
Interval = 0.25
dInterval = fSpeed*Interval
aInterval = aSpeed*Interval
###################
x,y=500,500
a=0
yaw = 0
points= []

kp.init()
me = tello.Tello()

me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()
me.takeoff()
me.move_up(80)


def getKeyBoardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 15
    aspeed = 50
    d = 0
    global yaw, x, y, a

    if kp.getKey("LEFT") :
         lr = -speed
         d = dInterval
         a = -180
    elif kp.getKey("RIGHT") :
         lr = speed
         d = -dInterval
         a = 180


    if kp.getKey("UP") :
         fb = speed
         d = dInterval
         a = 270

    elif kp.getKey("DOWN") : 
        fb = -speed
        d = -dInterval
        a = -90

    if kp.getKey("w") : ud = speed
    elif kp.getKey("s") : up = -speed

    if kp.getKey("a"):
         yv = aspeed
         yaw -= aInterval
    if kp.getKey("d"):
         yv = -aspeed
         yaw+=aInterval 

    if kp.getKey("q"):  me.land()
    if kp.getKey("t"): me.takeoff()
    sleep(Interval)

    a+=yaw
    x+=int(d*math.cos(math.radians(a)))
    y+=int(d*math.sin(math.radians(a)))




    return [lr, fb, ud, yv ,x , y]

def drawPoints(img , points):
    for point in points:
        cv2.circle(img,(point[0],point[1]),5,(0,0,255),cv2.FILLED)
    cv2.circle(img,(points[-1]),5,(255,0,255),cv2.FILLED)
    cv2.putText(img, f'({(points[-1][0] - 500)/100},{(points[-1][1]-500)/100})m',(points[-1][0]+10,points[-1][1]+30),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),1)




# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
detector = FaceDetector(minDetectionCon=0.7)
# _,img = cap.read()
# hi,wi =480,640
# hi , wi = img.shape[0],img.shape[1]


while True:
    # _,img = cap.read()
    img = me.get_frame_read().frame
    vals = getKeyBoardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    img ,objlist = detector.findFaces(img , draw=True)
    person = len(objlist)
    cv2.putText(img,f"Total person count : {person}",(5,30),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(255,0,255),1)
    img1 = np.zeros((1000,1000,3),np.uint8)
    points.append((vals[4],vals[5]))
    drawPoints(img1 , points)
    cv2.imshow("OUTPUT1",img1)
    

    
    # imgflip=cv2.flip(img,1)
    cv2.imshow('OUTPUT',img)

   
    if cv2.waitKey(5) & 0xff==('q'):
        # me.land()
        break
cv2.destroyAllWindows   
        

