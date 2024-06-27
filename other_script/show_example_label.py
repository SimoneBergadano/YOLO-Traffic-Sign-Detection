from tqdm import tqdm # serve per la progress bar
import random
import os
import shutil
import cv2

classNames = ["SpeedLimit 20 km/h", "SpeedLimit 30 km/h", "SpeedLimit 50 km/h", "SpeedLimit 60 km/h", "SpeedLimit 70 km/h", "SpeedLimit 80 km/h", "RestrictionEnds 80 km/h", "SpeedLimit 100 km/h", "SpeedLimit 120 km/h", "NoOvertaking", "NoOvertaking (trucks) ", "PriorityAtNextIntersection (danger)", "PriorityRoad", "GiveWay", "Stop", "NoTrafficBothWays", "NoTrucks", "NoEntry", "Danger", "BendLeft (danger)", "BendRight (danger)", "Bend (danger)", "UnevenRoad (danger)", "SlipperyRoad (danger)", "RoadNarrows (danger)", "Construction (danger)", "GeneralDanger", "PedestrianCrossing (danger)", "SchoolCrossing (danger)", "CyclesCrossing (danger)", "Snow (danger)", "Animals (danger)", "RestrictionEnds", "GoRight (mandatory)", "GoLeft (mandatory)", "GoStraight (mandatory)", "GoRightOrStraight (mandatory)", "GoLeftOrStraight (mandatory)", "KeepRight (mandatory)", "KeepLeft (mandatory)", "Roundabout (mandatory)", "RestrictionEnds (overtaking)", "RestrictionEnds (overtaking trucks)", "green trafficlight", "red trafficlight", "yellow trafficlight"]

classNamesITA = ["limite velocità 20kmh", "limite velocità 30kmh", "limite velocità 50kmh", "limite velocità 60kmh", "limite velocità 70kmh", "limite velocità 80kmh", "Fine restrizione 80kmh", "limite velocità 100kmh", "limite velocità 120kmh", "sorpasso vietato", "sorpasso vietato (Camion)", "dare precedenza al prossimo incrocio", "strada con priorità", "dare precedenza", "stop", "accesso vietato", "vietato (camion)", "accesso vietato", "pericolo generico", "curva sinistra", "curva destra", "curva", "strada sconnessa", "strada scivolosa", "restringimento carreggiata", "lavori in corso", "pericolo", "attraversamento pedonale", "attraversamento scolari", "attraversamento bici", "neve", "animali", "fine restrizione", "obbligo svoltare destra", "obbligo svoltare sinistra", "obbligo proseguire dritti", "procedere dritto o destra", "procedere dritto o a sinistra", "tenere la destra", "tenere la sinistra", "rotatoria", "fine restrizione sorpasso", "fine restrizione sorpasso (camion)", "semaforo verde", "semaforo rosso", "semaforo giallo"]


def readCorrectLabel(file_path, w_img, h_img): 

  prediction = []
  
  with open(file_path, 'r') as f:

    file_content = f.read()
    for line in file_content.split("\n"):
      if(len(line)==0):
        continue
      (cls, x, y, w, h) = line.split(" ")
      x, y, w, h = float(x), float(y), float(w), float(h)
      x, y, w, h = int(x*w_img), int(y*h_img), int(w*w_img), int(h*h_img)
      x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
      cls = int(cls)
      label = (cls, x1, y1, x2, y2)
      prediction.append(label)
      

    return prediction


yolo_dataset_path = "GTSDB_Yolo_trafficlight"

if os.path.isdir(yolo_dataset_path+"/images_bb"):
  shutil.rmtree(yolo_dataset_path+"/images_bb")


os.mkdir(yolo_dataset_path+"/images_bb")
os.mkdir(yolo_dataset_path+"/images_bb/train")
os.mkdir(yolo_dataset_path+"/images_bb/val")
os.mkdir(yolo_dataset_path+"/images_bb/test")

folder_path = yolo_dataset_path+"/images/train"
for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                    img = cv2.imread(file_path)
                    h, w, c = img.shape
                    path = (folder_path+"/"+file_name).replace("images", "labels")
                    path = path.replace(".jpeg", ".txt")
                    labels = readCorrectLabel(path, w, h)
                    for label in labels:
                        c, x1, y1, x2, y2 = label
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = classNamesITA[int(c)].replace("à", "a'")
                        cv2.putText(img, text, (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    path = file_path.replace("images", "images_bb")
                    cv2.imwrite(path, img)




folder_path = yolo_dataset_path+"/images/val"
for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                    img = cv2.imread(file_path)
                    h, w, c = img.shape
                    path = (folder_path+"/"+file_name).replace("images", "labels")
                    path = path.replace(".jpeg", ".txt")
                    labels = readCorrectLabel(path, w, h)
                    for label in labels:
                        c, x1, y1, x2, y2 = label
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = classNamesITA[int(c)].replace("à", "a'")
                        cv2.putText(img, text, (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    path = file_path.replace("images", "images_bb")
                    cv2.imwrite(path, img)

folder_path = yolo_dataset_path+"/images/test"
for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                    img = cv2.imread(file_path)
                    h, w, c = img.shape
                    path = (folder_path+"/"+file_name).replace("images", "labels")
                    path = path.replace(".jpeg", ".txt")
                    labels = readCorrectLabel(path, w, h)
                    for label in labels:
                        c, x1, y1, x2, y2 = label
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text = classNamesITA[int(c)].replace("à", "a'")
                        cv2.putText(img, text, (x2, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    path = file_path.replace("images", "images_bb")
                    cv2.imwrite(path, img)