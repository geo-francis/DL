from ultralytics import YOLO
import cv2
import cvzone
import math
# model=YOLO("../yolo_weights/yolov8n.pt")
# results=model("test_set\cats\cat.4002.jpg",show=True)
# cv2.waitKey(0)

## using web cam

cap=cv2.VideoCapture(0)
cap.set(3,640)## 3-- property id-here it is width of the image,1280 - width  
cap.set(4,480)##4-- property id-here it is height of the image,720 - height    
model=YOLO("../yolo_weights/yolov8m.pt")##model that detects the object in the image n-nano,m-medium,l-large
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
               'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
               'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

len(class_names)


while True:
    success,img=cap.read()## captures and stores it in the variable img, success variable indicates whether the frame was successfully read or not.
    results=model(img,stream=True)
    results=model(img,show=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            print("box is",box)
            ## used in opencv (Bounding box)
            x1,y1,x2,y2=box.xyxy[0]##extracts the coordinates of the top-left (x1, y1) and bottom-right (x2, y2) corners of the bounding box.
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)#converts it from tensore to int
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,255),3)##draws rectangle over the boxes with a color and thickness.
            ## used in cvzone
            w,h=x2-x1,y2-y1## w gives the width and h for height, the rest gives x,y the coordinates
            cvzone.cornerRect(img,(x1,y1,w,h))
            # confidence
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            special_name=class_names[cls]
            print("class_name is",cls)
            if special_name=='cell phone' and conf>0.5:
             cvzone.putTextRect(img, f'{special_name} {conf}', (min(0, x1), min(35,y1)))
    cv2.imshow("Image",img)
    cv2.waitKey(1)

