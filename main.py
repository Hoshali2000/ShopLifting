import cv2 as cv
import numpy as np


BODY_PARTS = {
    "Nose": 0,
    "Neck": 1,
    "RShoulder": 2,
    "RElbow": 3,
    "RWrist": 4,
    "LShoulder": 5,
    "LElbow": 6,
    "LWrist": 7,
    "RHip": 8,
    "RKnee": 9,
    "RAnkle": 10,
    "LHip": 11,
    "LKnee": 12,
    "LAnkle": 13,
    "REye": 14,
    "LEye": 15,
    "REar": 16,
    "LEar": 17,
    "Background": 18,
}

POSE_PAIRS = [
    ["Neck", "RShoulder"],
    ["Neck", "LShoulder"],
    ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"],
    ["LElbow", "LWrist"],
    ["Neck", "RHip"],
    ["RHip", "RKnee"],
    ["RKnee", "RAnkle"],
    ["Neck", "LHip"],
    ["LHip", "LKnee"],
    ["LKnee", "LAnkle"],
    ["Neck", "Nose"],
    ["Nose", "REye"],
    ["REye", "REar"],
    ["Nose", "LEye"],
    ["LEye", "LEar"],
]

inWidth = 360
inHeight = 360

object_detection = cv.dnn.DetectionModel("frozen_inference_graph.pb", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

object_detection.setInputSize(320, 320)
object_detection.setInputScale(1.0/127.5) 
object_detection.setInputMean((127.5, 127.5, 127.5)) 
object_detection.setInputSwapRB(True)

classlabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classlabels = fpt.read().rstrip('\n').split('\n')

cap = cv.VideoCapture("Copy of market.mp4")

font_scale = 3
font = cv.FONT_HERSHEY_PLAIN


cap = cv.VideoCapture("Copy of market.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output = cv.VideoWriter(
    "output_video.mp4", cv.VideoWriter_fourcc(*"mpv4"), 30, (frame_width, frame_height)
)

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    ClassIndex, confidence, box = object_detection.detect(frame, confThreshold=0.55)
    
    print(ClassIndex)
    if(len(ClassIndex)!= 0):
            for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), box):
                if(ClassInd <= 80):
                    cv.rectangle(frame, boxes, (255, 0, 0), 2)
                    cv.putText(frame, classlabels[ClassInd - 1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale,color=(0, 255, 0))

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(
        cv.dnn.blobFromImage(
            frame,
            1.0,
            (inWidth, inHeight),
            (127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
    )
    out = net.forward()
    out = out[
        :, :19, :, :
    ]  

    assert len(BODY_PARTS) == out.shape[1]

    points = []
    for i in range(len(BODY_PARTS)):
      
        heatMap = out[0, i, :, :]

       
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
      
        points.append((int(x), int(y)) if conf > 0.4 else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert partFrom in BODY_PARTS
        assert partTo in BODY_PARTS

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
    output.write(frame)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(
        frame, "%.2fms" % (t / freq), (10, 20), font, 0.5, (0, 0, 0)
    )

    cv.imshow("OpenPose using OpenCV", frame)

cap.release()
output.release()
cv.destroyAllWindows()
