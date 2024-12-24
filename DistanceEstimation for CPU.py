import cv2 
import pyttsx3


KNOWN_DISTANCE = 45 
PERSON_WIDTH = 16
MOBILE_WIDTH = 3.0


CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3


COLORS = [(255,0,0),(255,0,255),(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN =(0,255,0)
BLACK =(0,0,0)

FONTS = cv2.FONT_HERSHEY_COMPLEX


class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

yoloNet = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yoloNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

model = cv2.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    data_list =[]
    for (classid, score, box) in zip(classes, scores, boxes):
        
        color= COLORS[int(classid) % len(COLORS)]
    
        label = "%s : %f" % (class_names[classid], score)

        
        cv2.rectangle(image, box, color, 2)
        cv2.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)
    
        
        if classid == 0: 
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 67:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 2:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 3:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 7:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 1:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 5:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 10:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 16:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 39:
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        else:
            data_list.append(['Unknown', box[2], (box[0], box[1]-2)])

    return data_list



def focal_length_finder (measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length

 
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


ref_person = cv2.imread('ReferenceImages/image14.png')
ref_mobile = cv2.imread('ReferenceImages/image4.png')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[1][1]

person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]

print(f"Person width in pixels : {person_width_in_rf} mobile width in pixel: {mobile_width_in_rf}")


focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)

focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
cap = cv2.VideoCapture(0)


def read_annotations_aloud(annotations):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  

    for annotation in annotations:
        class_name = annotation["class_name"]
        distance = annotation["distance"]
        bbox = annotation["bounding_box"]
        x1, y1, x2, y2 = bbox

        text = f"{class_name}, distance: {distance:.2f} inches."
        engine.say(text)
        engine.runAndWait()

while True:
    ret, frame = cap.read()

    
    
    data = object_detector(frame) 
    annotations = []
    for d in data:
        if d[0] =='person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
             
            annotations.append({
            "class_name": d[0],
            "distance": distance,
            "bounding_box": (x, y, x + 150, y + 23)
            })
        elif d[0] =='cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
             
            annotations.append({
            "class_name": d[0],
            "distance": distance,
            "bounding_box": (x, y, x + 150, y + 23)
            })
        elif d[0] == 'bicycle':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
             
            annotations.append({
            "class_name": d[0],
            "distance": distance,
            "bounding_box": (x, y, x + 150, y + 23)
            })
        elif d[0] =='car':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
             
            annotations.append({
            "class_name": d[0],
            "distance": distance,
            "bounding_box": (x, y, x + 150, y + 23)
            })
        elif d[0] =='motorbike':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
             
            annotations.append({
            "class_name": d[0],
            "distance": distance,
            "bounding_box": (x, y, x + 150, y + 23)
            })
        elif d[0] =='bus':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
             
            annotations.append({
            "class_name": d[0],
            "distance": distance,
            "bounding_box": (x, y, x + 150, y + 23)
            })
        elif d[0] =='dog':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
             
            annotations.append({
            "class_name": d[0],
            "distance": distance,
            "bounding_box": (x, y, x + 150, y + 23)
            })
        elif d[0] =='truck':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
             
            annotations.append({
            "class_name": d[0],
            "distance": distance,
            "bounding_box": (x, y, x + 150, y + 23)
            })
        else:
            
            distance = distance_finder(100, 10, d[1])  
            x, y = d[2]
            
            annotations.append({
            "class_name": "Unknown",
            "distance": distance,
            "bounding_box": (x, y, x + 150, y + 23)
            })
        
       
        cv2.rectangle(frame, (x, y-3), (x+150, y+23), BLACK, -1)
        cv2.putText(frame, f'Dis: {round(distance, 2)} inch', (x+5, y+13), FONTS, 0.48, GREEN, 2)
    
    cv2.imshow('frame', frame)
    
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    read_annotations_aloud(annotations)
    
cv2.destroyAllWindows()
cap.release()
