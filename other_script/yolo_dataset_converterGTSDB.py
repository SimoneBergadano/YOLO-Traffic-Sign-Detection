from tqdm import tqdm # serve per la progress bar
import random
import os
import shutil
import cv2

dataset_path = "FullIJCNN2013"
yolo_dataset_path = "GTSDB_Yolo"

def main():
    
    if os.path.isdir(yolo_dataset_path):
        print(f"\n La cartella \"{yolo_dataset_path}\" esiste già, cancellarla se si vuole ricreare il dataset")
        return 1
    
    
    if not os.path.isdir(dataset_path):
        print(f"\n La cartella \"{dataset_path}\" che deve contenere il dataset di partenza non esiste")
        return 1

    os.mkdir(yolo_dataset_path)
    os.mkdir(yolo_dataset_path+"/images")
    os.mkdir(yolo_dataset_path+"/labels")

    os.mkdir(yolo_dataset_path+"/images/train")
    os.mkdir(yolo_dataset_path+"/images/val")
    os.mkdir(yolo_dataset_path+"/images/test")

    os.mkdir(yolo_dataset_path+"/labels/train")
    os.mkdir(yolo_dataset_path+"/labels/val")
    os.mkdir(yolo_dataset_path+"/labels/test")
    
    

    N = 900

    samples_resolution = []

    print("\n Conversione immagini: \n")
    for i in tqdm(range(N)):
        img = cv2.imread(dataset_path+"/"+f"{i:05d}.ppm")
        h, w, c = img.shape
        samples_resolution.append((h, w, c))
        cv2.imwrite(yolo_dataset_path+"/"+f"{i:05d}.jpeg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])

    print("\n Lettura e conversione bounding box: \n")
    with open(dataset_path+"/gt.txt", 'r') as file:
        file_content = file.read()
    labels = dict()
    for line in tqdm(file_content.split("\n")):
        if len(line)==0:
            continue
        fields = line.split(";")
        sample = fields[0].replace(".ppm", "")
        sample = int(sample)
        
        if sample not in labels:
            labels[sample]=[]
        (x1, y1, x2, y2, c) = tuple(fields[1:6])
        (c, x1, y1, x2, y2) = int(c), int(x1), int(y1), int(x2), int(y2)       
        (h_img, w_img, _) = samples_resolution[sample]
        (xc, yc, w, h) = (x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1
        (xc, yc, w, h) = (xc/w_img, yc/h_img, w/w_img, h/h_img)

        """
        xc, yc, w, h = int(xc*w_img), int(yc*h_img), int(w*w_img), int(h*h_img)
        x1, y1, x2, y2 = int(xc-w/2), int(yc-h/2), int(xc+w/2), int(yc+h/2)
        img = cv2.imread(dataset_path+"/"+f"{sample:05d}.ppm")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(yolo_dataset_path+f"/{sample:05d}.jpg", img)
        """
        

        labels[sample].append( (c, xc, yc, w, h) )

    print("\n Riorganizzazione cartelle e scrittura label nel formato yolo: \n")
    random_ord = list(range(N))
    random.shuffle(random_ord)
    for i in tqdm(range(N)):
        j = random_ord[i]
        if i<(0.8*N):
            images_dest_path = yolo_dataset_path+"/images/train"
            labels_dest_path = yolo_dataset_path+"/labels/train"
        elif i<(0.9*N):
            images_dest_path = yolo_dataset_path+"/images/val"
            labels_dest_path = yolo_dataset_path+"/labels/val"
        else:
            images_dest_path = yolo_dataset_path+"/images/test"
            labels_dest_path = yolo_dataset_path+"/labels/test"

        sample_name = f"{j:05d}"

        shutil.move(yolo_dataset_path+"/"+sample_name+".jpeg", images_dest_path)


        with open(labels_dest_path+"/"+sample_name+".txt", 'w') as file:
            # alcune immagini non contengono segnaletica
            if j in labels:
                for label in labels[j]:
                    file.write(f'{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n')
    
    with open(yolo_dataset_path+'/data.yaml', 'w') as f:
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("\n")
        f.write("nc: 43\n")
        f.write('names: ["SpeedLimit 20 km/h", "SpeedLimit 30 km/h", "SpeedLimit 50 km/h", "SpeedLimit 60 km/h", "SpeedLimit 70 km/h", "SpeedLimit 80 km/h", "RestrictionEnds 80 km/h", "SpeedLimit 100 km/h", "SpeedLimit 120 km/h", "NoOvertaking", "NoOvertaking (trucks) ", "PriorityAtNextIntersection (danger)", "PriorityRoad", "GiveWay", "Stop", "NoTrafficBothWays", "NoTrucks", "NoEntry", "Danger", "BendLeft (danger)", "BendRight (danger)", "Bend (danger)", "UnevenRoad (danger)", "SlipperyRoad (danger)", "RoadNarrows (danger)", "Construction (danger)", "GeneralDanger", "PedestrianCrossing (danger)", "SchoolCrossing (danger)", "CyclesCrossing (danger)", "Snow (danger)", "Animals (danger)", "RestrictionEnds", "GoRight (mandatory)", "GoLeft (mandatory)", "GoStraight (mandatory)", "GoRightOrStraight (mandatory)", "GoLeftOrStraight (mandatory)", "KeepRight (mandatory)", "KeepLeft (mandatory)", "Roundabout (mandatory)", "RestrictionEnds (overtaking)", "RestrictionEnds (overtaking trucks)"]\n')



    return 0

main()

