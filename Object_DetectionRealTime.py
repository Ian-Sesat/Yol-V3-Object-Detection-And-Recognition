#Importing the necessary libraries:
import cv2
import numpy as np
#reading the deep Neural network from the gotten weights and configuration:
net=cv2.dnn.readNet('yolov3_training_last.weights', 'yolov3_testing.cfg')
#getting the classes Names from the classes.txt file:
classes=[]
with open('classes.txt ','r') as f:
    classes=f.read().splitlines()
print(classes)
#Getting the video to be analyzed in realtime:

cam=cv2.VideoCapture(0)   
height=720
width=1080
cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

while True:
    ignore, frame=cam.read()
    #Getting a blob out of our frame:
    blob=cv2.dnn.blobFromImage(frame,1/255,(416,416),(0,0,0),swapRB=True, crop=False)

    #Inputing our blob to the neural network and getting our output out of the neural network:
    net.setInput(blob)
    output_layer_names=net.getUnconnectedOutLayersNames()
    layerOutputs=net.forward(output_layer_names)

    #Setting up and getting the bounding boxes and parameters for the object detection:
    boxes=[]
    confidences=[]
    class_ids=[]

    #Getting the bounding boxes, scores and class_ids for object detection
    for output in layerOutputs:
        for detection in output:
            scores=detection[5:]
            class_id=np.argmax(scores)
            confidence=scores[class_id]
            if confidence>0.6:
                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w= int(detection[2]*width)
                h= int(detection[3]*height)
                x= int(center_x-w/2)
                y= int(center_y-h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    #Filtering out of the unecessary bounding boxes:
    indexes=cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #Setting up the font and color for the text and bounding box respectively:
    font=cv2.FONT_HERSHEY_DUPLEX
    colors=np.random.uniform(0,255,size=(len(boxes),3))
    nutsArray=[]
    boxesArray=[]
    spannersArray=[]
    if len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h=boxes[i]
            label=str(classes[class_ids[i]])
            frameHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            myROI=frameHSV[x,y]
            myROIHue=myROI[0]
            #print(myROIHue)
            if label=='box':
                #print(myROI)
                if myROIHue >=0 and myROIHue <= 10 or myROIHue >=159 and myROIHue <=180:
                    boxColor='Red'
                    label='Red Box'
                elif myROIHue>10 and myROIHue<25:
                    boxColor='Orange'
                    label='Orange box'
                elif myROIHue>=25 and myROIHue <36:
                    boxColor='Yellow'
                    label='Yellow Box'
                elif myROIHue>=36 and myROIHue <90:
                    boxColor='Green'
                    label='Green Box'
                elif myROIHue>=90 and myROIHue <128:
                    boxColor='Blue'
                    label='Blue Box'
                else :
                    boxColor='Purple'
                    label='Purple Box'

                boxLabel='box'+str(boxColor)
                boxesArray.append(boxLabel)
                
            if label=='spanner':
                if myROIHue >=0 and myROIHue <= 10 or myROIHue >=159 and myROIHue <=180:
                    spannerColor='Red'
                    label='Red Spanner'
                elif myROIHue>10 and myROIHue<25:
                    spannerColor='Orange'
                    label='Orange Spanner'
                elif myROIHue>=25 and myROIHue <36:
                    spannerColor='Yellow'
                    label='Yellow Spanner'
                elif myROIHue>=36 and myROIHue <90:
                    spannerColor='Green'
                    label='Green Spanner'
                elif myROIHue>=90 and myROIHue <128:
                    spannerColor='Blue'
                    label='Blue Spanner'
                else :
                    spannerColor='Purple'
                    label='Purple Spanner'
                spannerLabel=str(spannerColor)+' Spanner'
                spannersArray.append(spannerLabel)

                
            if label=='nut':
                if myROIHue >=0 and myROIHue <= 10 or myROIHue >=159 and myROIHue <=180:
                    nutColor='Red'
                    label='Red Nut'
                elif myROIHue>10 and myROIHue<25:
                    nutColor='Orange'
                    label='Orange Nut'
                elif myROIHue>=25 and myROIHue <36:
                    nutColor='Yellow'
                    label='Yellow Nut'
                elif myROIHue>=36 and myROIHue <90:
                    nutColor='Green'
                    label='Green Nut'
                elif myROIHue>=90 and myROIHue <128:
                    nutColor='Blue'
                    label='Blue Nut'
                else :
                    nutColor='Purple'
                    label='Purple Nut'
                nutLabel='nut'+str(nutColor)
                nutsArray.append(nutLabel)

            confidence=str(round(confidences[i],2))#Rounds off the confidence to 2decimal places
            color=colors[i]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+' '+confidence,(x,y-20),font,1,(255,255,0),1)


    numBoxes=len(boxesArray)
    numSpanners=len(spannersArray)
    numNuts=len(nutsArray)
    #print('Boxes: ', numBoxes)
    #print('Spanners: ',numSpanners)
    #print('Nuts: ', numNuts)
    cv2.putText(frame,'Boxes: '+str(numBoxes),(width-200, height-40),font,1,(0,255,0),1)
    cv2.putText(frame,'Spanners: '+str(numSpanners),(width-200, height-20),font,1,(0,255,0),1)
    cv2.putText(frame,'Nuts: '+str(numNuts),(width-200, height),font,1,(0,255,0),1)
    cv2.imshow('My image',frame)
    if cv2.waitKey(1) &0xff== ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
