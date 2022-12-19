import tkinter

import cv2
import numpy as np
from tkinter import *

root = Tk()
root.title("OBJECT DETECTION IN IMAGES")
root.geometry('1500x1000')
#root.config(bg="6391BA")
bg = PhotoImage(file = "oop.png")
label1 = Label( root, image = bg)
label1.place(x = -200, y = -200)
label2 = Label( root, text = "HELLO USER!",font=("Rockwell",25),bg="darkcyan",fg="white")
label2.pack(pady = 50)
#l.pack()
#l.config(bg = "skyblue")    # These all are for making a new window


def V_IMAGE():
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()  # These all are upload yolo algorithim and creating and read a label
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    img = cv2.imread("koko.jpg")  # read a image
    # cap = cv2.VideoCapture('dog.jpg')
    # while True:
    # _, img = cap.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 225, (608, 608), (0, 0, 0), swapRB=True, crop=False, )
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []  # these all are variables of images detection
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 4))
    if len(indexes) > 0:  # the len of indexes in doesn't be in 0
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 4, (0, 255, 225), 9)
    dim = (900, 700)
    resize = cv2.resize(img, dim)
    cv2.imshow('op', resize)
    key = cv2.waitKey(0)
    if key == 81 or key == 113:
        cv2.destroyAllWindows()


def V_VIDEO():
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()  # These all are upload yolo algorithim and creating and read a label
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # img = cv2.imread('op.jpg')
    cap = cv2.VideoCapture('gop.mp4')  # video reading
    while True:
        success, frame = cap.read()
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 225, (608, 608), (0, 0, 0), swapRB=True, crop=False, )
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 4))
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (225, 255, 225), 3)
        dim = (900, 700)
        resize = cv2.resize(frame, dim)
        cv2.imshow('op', resize)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            cv2.destroyAllWindows()
            break


def V_REAL_TIME():
    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().splitlines()  # These all are upload yolo algorithim and creating and read a label
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # img = cv2.imread('op.jpg')
    cap = cv2.VideoCapture(0)  # main web cam reader
    while True:
        success, frame = cap.read()
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 225, (608, 608), (0, 0, 0), swapRB=True, crop=False, )
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.6)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 4))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (225, 255, 225), 2)
        dim = (500, 500)
        resize = cv2.resize(frame, dim)
        cv2.imshow('op', resize)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            cv2.destroyAllWindows()
            break


def S_IMAGE():
    classes = []
    with open('soso.names', 'r') as f:
        classes = f.read().splitlines()  # These all are upload yolo algorithim and creating and read a label
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    img = cv2.imread('op.jpg')  # read a image
    # cap = cv2.VideoCapture('dog.jpg')
    # while True:
    # _, img = cap.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 225, (608, 608), (0, 0, 0), swapRB=True, crop=False, )
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []  # these all are variables of images detection
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 4))
    if len(indexes) > 0:  # the len of indexes in doesn't be in 0
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 4, (0, 255, 225), 5)
    dim = (900, 700)
    resize = cv2.resize(img, dim)
    cv2.imshow('op', resize)
    key = cv2.waitKey(0)
    if key == 81 or key == 113:
        cv2.destroyAllWindows()

def S_VIDEO():
    classes = []
    with open('soso.names', 'r') as f:
        classes = f.read().splitlines()  # These all are upload yolo algorithim and creating and read a label
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # img = cv2.imread('op.jpg')
    cap = cv2.VideoCapture('gop.mp4')  # video reading
    while True:
        success, frame = cap.read()
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 225, (608, 608), (0, 0, 0), swapRB=True, crop=False, )
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 4))
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (225, 255, 225), 3)
        dim = (900, 700)
        resize = cv2.resize(frame, dim)
        cv2.imshow('op', resize)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            cv2.destroyAllWindows()
            break

def S_REAL_TIME():
    classes = []
    with open('soso.names', 'r') as f:
        classes = f.read().splitlines()  # These all are upload yolo algorithim and creating and read a label
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # img = cv2.imread('op.jpg')
    cap = cv2.VideoCapture(0)  # main web cam reader
    while True:
        success, frame = cap.read()
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 225, (608, 608), (0, 0, 0), swapRB=True, crop=False, )
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.6)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 4))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (225, 255, 225), 2)
        dim = (500, 500)
        resize = cv2.resize(frame, dim)
        cv2.imshow('op', resize)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            cv2.destroyAllWindows()
            break

def R_IMAGE():
    classes = []
    with open('roro.names', 'r') as f:
        classes = f.read().splitlines()  # These all are upload yolo algorithim and creating and read a label
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    img = cv2.imread('op.jpg')  # read a image
    # cap = cv2.VideoCapture('dog.jpg')
    # while True:
    # _, img = cap.read()
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 225, (608, 608), (0, 0, 0), swapRB=True, crop=False, )
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []  # these all are variables of images detection
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 4))
    if len(indexes) > 0:  # The len of indexes in doesn't be in 0
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 4, (0, 255, 225), 5)
    dim = (900, 700)
    resize = cv2.resize(img, dim)
    cv2.imshow('op', resize)
    key = cv2.waitKey(0)
    if key == 81 or key == 113:
        cv2.destroyAllWindows()

def R_VIDEO():
    classes = []
    with open('roro.names', 'r') as f:
        classes = f.read().splitlines()  # These all are upload yolo algorithim and creating and read a label
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # img = cv2.imread('op.jpg')
    cap = cv2.VideoCapture('gop.mp4')  # video reading
    while True:
        success, frame = cap.read()
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 225, (608, 608), (0, 0, 0), swapRB=True, crop=False, )
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 4))
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (225, 255, 225), 3)
        dim = (900, 700)
        resize = cv2.resize(frame, dim)
        cv2.imshow('op', resize)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            cv2.destroyAllWindows()
            break

def R_REAL_TIME():
    classes = []
    with open('roro.names', 'r') as f:
        classes = f.read().splitlines()  # These all are upload yolo algorithim and creating and read a label
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # img = cv2.imread('op.jpg')
    cap = cv2.VideoCapture(0)  # main web cam reader
    while True:
        success, frame = cap.read()
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1 / 225, (608, 608), (0, 0, 0), swapRB=True, crop=False, )
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.6)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 4))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                color = colors[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
                cv2.putText(frame, label + " " + confidence, (x, y + 20), font, 2, (225, 255, 225), 2)
        dim = (500, 500)
        resize = cv2.resize(frame, dim)
        cv2.imshow('op', resize)
        key = cv2.waitKey(1)
        if key == 81 or key == 113:
            cv2.destroyAllWindows()
            break

def exi():
    quit()


def button():
    root1 = Tk()
    root1.geometry('1500x1000')
    root1.config(bg="#48D1CC")
    l = Label(root1, text='VEHICLE DETECTION', font=("Rockwell", 25), fg="white", bg="darkcyan")
    l.pack()
    b = Button(root1, text='DETECT AN IMAGE', width=25, height=2, font=("Rockwell", 24),command=V_IMAGE, fg="white", bg="darkslategray")
    b2 = Button(root1, text='DETECT A VIDEO', width=25, height=2, font=("Rockwell", 24),command=V_VIDEO, fg="white", bg="darkslategray")
    b3 = Button(root1, text='DETECT A REAL TIME VIDEO',width=25, height=2, font=("Rockwell", 24), command=V_REAL_TIME, fg="white", bg="darkslategray")
    b4 = Button(root1, text='BACK',width=25, height=2, font=("Rockwell", 24), command=first,fg="white", bg="darkslategray")
    b.place(x=510, y=70)
    b2.place(x=510, y=180)
    b3.place(x=510, y=290)
    b4.place(x=510, y=500)
    root1.mainloop()

def button1():
    root1 = Tk()
    root1.geometry('1500x1000')
    root1.config(bg="#48D1CC")
    l = Label(root1, text='STRUCTURE DETECTION', font=("Rockwell",25), fg="white", bg="darkcyan")
    l.pack()
    b = Button(root1, text='DETECT AN IMAGE', width=25, height=2, font=("Rockwell", 24),command=S_IMAGE ,fg="white", bg="darkslategray")
    b2 = Button(root1, text='DETECT A VIDEO',width=25, height=2, font=("Rockwell", 24), command=S_VIDEO,fg="white", bg="darkslategray")
    b3 = Button(root1, text='DETECT A REAL TIME VIDEO',width=25, height=2, font=("Rockwell", 24), command=S_REAL_TIME,fg="white", bg="darkslategray")
    b4 = Button(root1, text='BACK',width=25, height=2, font=("Rockwell", 24), command=first,fg="white", bg="darkslategray")
    b.place(x=510, y=70)
    b2.place(x=510, y=180)
    b3.place(x=510, y=290)
    b4.place(x=510, y=500)
    root1.mainloop()

def button2():
    root = Tk()
    root.geometry('1500x1000')
    root.config(bg="#48D1CC")
    #mk = PhotoImage(file="2.png")
    l = Label(root, text='ROAD SIGNAGES DETECTION', font=("Rockwell",25), fg="white", bg="darkcyan")
    l.pack()
    b = Button(root, text='DETECT AN IMAGE', width=25, height=2, font=("Rockwell", 24), command=R_IMAGE, fg="white", bg="darkslategray")
    b.place(x=510, y=70)
    b2 = Button(root, text='DETECT A VIDEO',width=25, height=2, font=("Rockwell", 24), command=R_VIDEO,fg="white", bg="darkslategray")
    b2.place(x=510, y=180)
    b3 = Button(root, text='DETECT A REAL TIME VIDEO',width=25, height=2, font=("Rockwell", 24), command=R_REAL_TIME,fg="white", bg="darkslategray")
    b3.place(x=510, y=290)
    b4 = Button(root,text='BACK',width=25, height=2, font=("Rockwell", 24), command=first,fg="white", bg="darkslategray")
    b4.place(x=510, y=500)
    root.mainloop()

def info():
    root = Tk()
    root.geometry('1500x1000')
    root.config(bg="#48D1CC")
    mk = PhotoImage(file="2.png")
    l = Label(root, text='DEVELOPERS TERM', font=("Rockwell", 25), fg="white", bg="darkcyan")
    l.pack()
    b1 = Button(root,text='back',width=20, height=1,font =("Rockwell",24),command=first,fg="white",bg="darkcyan")
    b1.place(x=100, y=700)
    root.mainloop()

def first():

    root = Tk()
    root.geometry('1500x1000')
    root.config(bg="#48D1CC")
    mk = PhotoImage(file="2.png")
    l = Label(root, text='OBJECT DETECTION',font=("Rockwell",25),fg = "white",bg="darkcyan")
    l.pack()

    #l1.place(x=-200, y=-200)
    m = Button(root, text='VECHILES', width=25, height=2, font=("Rockwell", 24), command=button, fg="white", bg="darkslategray")
    m.place(x=510, y=70)
    m = Button(root, text='STRUCTURES', width=25, height=2, font=("Rockwell", 24), command=button1, fg="white", bg="darkslategray")
    m.place(x=510, y=180)
    m = Button(root, text='ROAD SIGNAGES', width=25, height=2, font=("Rockwell", 24), command=button2, fg="white", bg="darkslategray")
    m.place(x=510, y=290)
    b1 = Button(root, text='BACK', width=25, height=2, font=("Rockwell", 24), command=exit, fg="white", bg="darkslategray")
    b1.place(x=510, y=500)
    b2 = Button(root, text='Developers term',width= 20,height = 1, command=info,fg="white", bg = "darkslategray")
    b2.place(x=1000, y=700)
    root.mainloop()

b2 = Button(root, text='GET STARTED',width = 25,height = 2, font = ("Rockwell", 24),command=first ,fg="white", bg = "darkslategray")
b2.place(x=175,y=225)
b1 = Button(root, text='EXIT',width = 25,height = 2, font = ("Rockwell", 24),command=exi ,fg="white", bg = "darkslategray")
b1.place(x=175,y=500)

root.mainloop()
