# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 17:27:23 2021

@author: Manan doshi
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt 


#getting the image 
#img_read=cv2.imread('images/scene2.jpg')
video=cv2.VideoCapture('videos/video_sample4.mp4')
#webcam=cv2.VideoCapture(0)
while (video.isOpened()):
    ret,current_frame=video.read()
    img_read=current_frame
        
    img_height=img_read.shape[0]
    img_width=img_read.shape[1]
    plt.imshow(img_read)
    print(img_height,img_width)
    
    cv2.imshow('input image',img_read)
    
    
    class_id_lists=[]
    boxes_list=[]
    confidence_list=[]
    
    
    
    #this will convert and image in to the blob as yolo require bob image because every yolo model has specified resolution of blob image specified to it
    #blob method is used to convert the evry pixel of an image to in the range 0-1 then pass it to yolo model
    img_blob=cv2.dnn.blobFromImage(img_read,1/255,(416,416),swapRB=True,crop=False)
    i=img_blob[0].reshape(416,416,3)
    plt.imshow(i)
    print("binary_object_image: {}".format(img_blob.shape))
    
    
    
    #some predefined classes in Yolo
    class_labels = ["prohibitory","danger","mandatory","other"]
    
    
    #creating array of the colors of the bounding box
    class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors,(16,1))
    # tiling is method of conituous coloring of the bounding box around that object
    
    
    
    
    yolo_model=cv2.dnn.readNetFromDarknet('model/custom_yolov3.cfg','model/custom_yolov3_14000.weights')
    #loading the model and its weights 
    
    
    yolo_layers=yolo_model.getLayerNames()
    yolo_output_layer=[yolo_layers[yolo_layer[0]-1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]
    print(yolo_output_layer)
    #YOLOv3 has 3 output layers (82(large object detection), 94(medium object detection) and 106(small object detection)) as the figure shows.
    #getLayerNames(): Get the name of all layers of the network.
    #getUnconnectedOutLayers(): Get the index of the output layers.
    #These two functions are used for getting the output layers (82,94,106).
    
    yolo_model.setInput(img_blob)
    object_detection_layers=yolo_model.forward(yolo_output_layer)
    print(object_detection_layers)
    
    
    
    
    #82 layer aaya mera object_detection_layers me usko me object_detection_layer me dala phir us 82 me object_detection kya hua mtlb ek array 
    #array contain mera fullimage height and weidth ,object in that image ka height and width ,plus us particluar grid me wo object hai ki nai
    #uska probabilty ,last but least me sare class ka mtlb 80 class ka name mtlb in short jo apna labelcell jo array hai usko me read karai forloopse
    #wobhi 82th layer ka nai sare unconnected layers ka . 
    for object_detection_layer in object_detection_layers:
        for object_detection in object_detection_layer:
            
            all_scores=object_detection[5:]
            
            predicted_class_id=np.argmax(all_scores)
         
           
            prediction_confidence=all_scores[predicted_class_id]
           #all_scores me mera pahila 5 array element chhodke bake ke class ka probability dega
           #predicted_class_id mera usme me se highest probability  of 80 class jo hai wo dega in short 0 wale ko hataega
           #prediction_confidence me me wo probabilities assign krega
           #phir agr wo highest probabilty mera 0.20 se jyada hoga to me bounding box banana chalu krega
     #allscores = sare classeske probability aaenge that is 0-79
    #predicted_class_id= wo scores honge jinka probability 0 nai hoga plus unka index hoga
    #prediction_confidence= me mera wo sare sirf probability  values honge       
           #ye jo 0.20 hai wo mje aisa choose karna padega jab mera sensitivity or specificity of an image ka variance kam hoga 
           #ye sirf trail an error se hi niklega i this case mera 0.20 threshold barabar ho raha hai
            if prediction_confidence>0.20:
               
                
                predicted_class_label=class_labels[predicted_class_id]
                bounding_box=object_detection[0:4]*np.array([img_width,img_height,img_width,img_height])
                #in this bounding box there comes four coordinate of object in the image
                #center_x=int(object_detection[0]*width)=box_center_x_pt
                #center_x=int(object_detection[0]*height)=box_center_y_pt
                #w=int(object_detection[0]*width)=box_width
                #h=int(object_detection[0]*height)=box_height
                #x=int(center_x-w/2)=start_x_pt & end_y_pt
                #y=int(center_x-h/2)=start_y_pt & end_x_pt
                print(bounding_box)
                print(prediction_confidence)
                predicted_class_label="{}:{:.2f}%".format(predicted_class_label,prediction_confidence*100)
                print("predicted object {}".format(predicted_class_label))
                (box_center_x_pt,box_center_y_pt,box_width,box_height)=bounding_box.astype("int")
                #wo jo 4 mje mile hai usme se pahila wala -secondlastwala/2 karega to start x point milega
                start_x_pt=int(box_center_x_pt-(box_width/2))
                print(start_x_pt)
                #wo jo 4 mje mile hai usme se second wala -lastwala/2 karega to start y point milega
                start_y_pt=int(box_center_y_pt-(box_height/2))
                print(start_y_pt)
                
                class_id_lists.append(predicted_class_id)
                # wo scores honge jinka probability 0 nai hoga plus unka index hoga append krunga me
                
                confidence_list.append(float(prediction_confidence))
                #me mera wo sare sirf probability  values honge list me
                
                boxes_list.append([start_x_pt,start_y_pt,int(box_width),int(box_height)])
    print(class_id_lists)
    print(confidence_list)
    print(boxes_list)            
               
                
    # Applying the NMS will return only the selected max value ids while suppressing the non maximum (weak) overlapping bounding boxes      
    # Non-Maxima Suppression confidence set as 0.5 & max_suppression threhold for NMS as 0.4 (adjust and try for better perfomance)
    
    #yaha pe mera non max supression algorithm kam krega 
    #jis boc ka threshold sabse jyada hai usko lega aur baki ke box ko intersection over union krdega ye algorithm
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.5, 0.4)
    for max_valueid in max_value_ids:
        max_class_id = max_valueid[0]
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]
        #get the predicted class id and label
        predicted_class_id = class_id_lists[max_class_id]
        print(predicted_class_id)
        predicted_class_label = class_labels[predicted_class_id]
        print(predicted_class_label)
        prediction_confidence = confidence_list[max_class_id]
        print(prediction_confidence)
                
                
                
        end_x_pt=start_x_pt+box_width
        end_y_pt=start_y_pt+box_height
        box_color=class_colors[predicted_class_id]
        box_color=[int(c) for c in box_color]
        predicted_class_label="{}:{:.2f}%".format(predicted_class_label,prediction_confidence*100)#class ka name,uska confidence into 100 for percentage
        print("predicted object {}".format(predicted_class_label))
        cv2.rectangle(img_read,(start_x_pt,start_y_pt),(end_x_pt,end_y_pt),box_color,1)
        cv2.putText(img_read,predicted_class_label,(start_x_pt,start_y_pt-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,1)
        
    cv2.imshow("Detection Output", img_read)
    print(len(boxes_list))
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video.release()
cv2.destroyAllWindows()

            
            
        