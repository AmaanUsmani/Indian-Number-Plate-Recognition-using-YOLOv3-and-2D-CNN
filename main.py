import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import shutil
# from cv2 import dnn

import util


# define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

input_dir1 = 'C:/Users/sayed/OneDrive/Documents/GitHub/F28PA/Dataset/google_images'
input_dir2 = 'C:/Users/sayed/OneDrive/Documents/GitHub/F28PA/Dataset/State-wise_OLX'
input_dir3 = 'C:/Users/sayed/OneDrive/Documents/GitHub/F28PA/Dataset/video_images'
destinationdir = 'C:/Users/sayed/OneDrive/Documents/GitHub/F28PA/NotDetectedNP'
# input_dir = 'C:/Users/amaan/OneDrive/Desktop/ANPR/yolov3-from-opencv-object-detection/testdata'

charRecogRate = 0
accuracy = 0
totalCount = 0
successCount = 0
count = 0 
actualLabelCount = 0
plateLocalizeCount = 0
plateLocalizeRate = 0
sumBBox = 0
len_bbox = 0
imgno = 0
temppc = 1
folderList = [input_dir1, input_dir2, input_dir3]
folderList1 = [input_dir1]
for folderName in folderList:
    for img_name in os.listdir(folderName):
        if img_name[-3:] == 'jpg' or img_name[-3:] == 'png' or img_name[-4:] == 'jpeg':
            imgno = imgno + 1
            img_path = os.path.join(folderName, img_name)
            print(img_path)
            # load class names
            with open(class_names_path, 'r') as f:
                class_names = [j[:-1] for j in f.readlines() if len(j) > 2]
                f.close()

            # load model
            net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

            # load image
            img = cv2.imread(img_path)

            # try:
            H, W, _ = img.shape
            # except:
            #     continue

            # convert image
            blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

            # get detections
            net.setInput(blob)

            detections = util.get_outputs(net)

            reader = easyocr.Reader(['en'], gpu=True)
            # bboxes, class_ids, confidences
            bboxes = []
            class_ids = []
            scores = []

            for detection in detections:
                # [x1, x2, x3, x4, x5, x6, ..., x85]
                bbox = detection[:4]

                xc, yc, w, h = bbox
                bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

                bbox_confidence = detection[4]

                class_id = np.argmax(detection[5:])
                score = np.amax(detection[5:])

                bboxes.append(bbox)
                class_ids.append(class_id)
                scores.append(score)

            # apply nms
            bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)
            
            if(len(bboxes)==0):
                dest = shutil.move(img_path, destinationdir)
                if img_path[-4:] == 'jpeg':
                    dest = shutil.move(img_path[:-5]+".xml", destinationdir)
                    # file1 = open(img_path[:-5]+".xml")              #checking for file names that end with .xml
                else:
                    dest = shutil.move(img_path[:-4]+".xml", destinationdir)
                    # file1 = open(img_path[:-4]+".xml")
                continue

            for bbox_, bbox in enumerate(bboxes):
                xc, yc, w, h = bbox

                license_plate = img[int(yc - (h / 2)):int(yc + (h / 2)), int(xc - (w / 2)):int(xc + (w / 2)), :].copy()

                img = cv2.rectangle(img,
                                    (int(xc - (w / 2)), int(yc - (h / 2))),
                                    (int(xc + (w / 2)), int(yc + (h / 2))),
                                    (0, 255, 0),
                                    10)

                license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                # blur = cv2.GaussianBlur(license_plate_gray, (5, 5), 0)
                _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

                contours, hierarchy = cv2.findContours(license_plate_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # if not os.path.exists('char_images2'):
                #     os.mkdir('char_images2')
                
                # # Loop over the contours and extract the characters
                # for i, contour in enumerate(contours):
                #     # Get the bounding box coordinates of the character           CODE FOR SEGMENTATION AND STORING
                #     (x, y, wi, ht) = cv2.boundingRect(contour)
                #     # Filter out contours that are too small or too large
                #     contour_area = cv2.contourArea(contour)
                #     if contour_area < 20 or contour_area > 1000:
                #         # print(f'Contour {i} filtered out: area = {contour_area}')
                #         continue
                #     # Filter out contours that have a width to height ratio that is too high or too low
                #     aspect_ratio = float(w)/h
                #     if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                #         # print(f'Contour {i} filtered out: aspect ratio = {aspect_ratio}')
                #         continue
                #     # Extract the character from the image and resize it to a fixed size
                #     char = cv2.resize(license_plate_thresh[y:y+ht, x:x+wi], (48, 48))
                #     # Save the character image to the char_images directory
                #     cv2.imwrite(f'char_images2/char2_{temppc}.png', char)
                #     temppc+=1

                output = reader.readtext(license_plate_thresh)
                final_text = ''
                text_score = 0
                for out in output:
                    text_bbox, text, text_score = out
                    # if text_score > 0.4:
                    text = text.replace(" ", "")        #removing all empty spaces from recognized text
                    s=""
                    final_text = final_text + text
                    final_text = final_text.upper()     #converting all recognized text to uppercase 
                    for i in final_text:                #removing all characters except letters and numbers from the recognized text
                        if ord(i) in range(48,58) or ord(i) in range(65,91) or ord(i) in range(97,123):
                            s+=i
                    final_text = s
                print(final_text, text_score)

                # if len(final_text) == 0:
                ## print(len(final_text))      

            # plt.figure()
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            fig = plt.figure()
            plt.imshow(cv2.cvtColor(license_plate_thresh, cv2.COLOR_BGR2RGB))
            # if(len(bboxes) > 0):
            #     fig.savefig('Cropped Number Plates/' + folderName[55:] + str(imgno) + '.png')
            # plt.show()
            plt.close()
            

            if img_name[-4:] == 'jpeg':                     
                file1 = open(img_path[:-5]+".xml")              #checking for file names that end with .xml
            else:
                file1 = open(img_path[:-4]+".xml")
    
            
            fileList = file1.readlines()
            

            for word in fileList:                           
                if '<name>' in word:
                    actualLabel = word[8:-8]                    #finding the name tag in the xml document to retrieve
                    print('actual label: '+ actualLabel)        #the number plate
                    totalCount = totalCount + 1
                    break
            
            if(len(bboxes)>=1):
                if (final_text in actualLabel):
                    count = count + len(final_text)

                elif (actualLabel in final_text):
                    count = count + len(actualLabel)
            
                else:
                    for i in range(len(actualLabel)):
                        if i < len(final_text):
                            if final_text[i] == actualLabel[i]:
                                count = count + 1

        
                actualLabelCount = actualLabelCount+ len(actualLabel)

            print('Accuracy per character: ', count/actualLabelCount)  
            len_bbox = len(bboxes)
            if len_bbox > 1:
                len_bbox = 1
            sumBBox = sumBBox + len_bbox                     
            print('Accuracy of number plates detected: ', str(sumBBox/totalCount))
                

            if final_text == actualLabel:
                # print("SUCCESS")                                
                successCount = successCount + 1                 #if the number plate recognized by the ocr is the same
                accuracy = round((successCount)/totalCount,4)   #as the number plate in the xml doc, print success
            else:
                # print("FAILURE")
                accuracy = round((successCount)/totalCount,4)
            ## print("Accuracy = " + str(accuracy))         ACCURACY OF CORRECTLY RECOGNIZED NUMBER PLATES
            print("Number of license plates tested " + str(totalCount))
            print("Img Number"+ str(imgno))
            print("------------------------------------------------------")
        
            file1.close
            
