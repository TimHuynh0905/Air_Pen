import numpy as np
import idx2numpy
import random
import cv2 as cv
import matplotlib.pyplot as plt
from collections import deque
from NN_Model import NN_Model

raw_X_train = idx2numpy.convert_from_file('/Users/timhuynh0905/Documents/Air_Pen/EMNIST_data/emnist-byclass-train-images-idx3-ubyte')
raw_y_train = idx2numpy.convert_from_file('/Users/timhuynh0905/Documents/Air_Pen/EMNIST_data/emnist-byclass-train-labels-idx1-ubyte')
raw_X_test = idx2numpy.convert_from_file('/Users/timhuynh0905/Documents/Air_Pen/EMNIST_data/emnist-byclass-test-images-idx3-ubyte')
raw_y_test = idx2numpy.convert_from_file('/Users/timhuynh0905/Documents/Air_Pen/EMNIST_data/emnist-byclass-test-labels-idx1-ubyte')

model= NN_Model(raw_X_train = raw_X_train,
                raw_y_train = raw_y_train)

model.train()
test_res = model.test(raw_X_test = raw_X_test, raw_y_test = raw_y_test)
print(test_res)

# index = random.randrange(len(raw_X_test))
# model.predict(raw_X_test[index]) 

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 490)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 640)
kernel = np.ones((5, 5), np.uint8)

lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

notepad = np.zeros((490,640,3), dtype=np.uint8)
digit = np.zeros((200, 200, 3), dtype=np.uint8)
cv.namedWindow('Note', cv.WINDOW_AUTOSIZE)

pts = deque(maxlen = 512)

result = "Null"

while(cap.isOpened()):
    _, img = cap.read()
    img = cv.flip(img, 1)
    # img = img.astype('float32')

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    mask = cv.erode(mask, kernel, iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.dilate(mask, kernel, iterations=1)
    
    # res = cv.bitwise_and(img,img, mask= mask)
    
    cnts, heir = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    center = None
    
    if len(cnts) > 0:
        cnt = max(cnts, key=cv.contourArea)

        ((x, y), radius) = cv.minEnclosingCircle(cnt)
        if radius > 5:
            cv.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)

        M = cv.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        
        pts.appendleft(center)
        for i in range(1,len(pts)):
            if pts[i-1]is None or pts[i] is None:
                continue
            cv.line(img, pts[i-1], pts[i], (0,0,225), 20)
            cv.line(notepad, pts[i-1], pts[i], (225,225,225), 20)
    elif len(cnts) == 0: 
        notepad = np.zeros((490,640,3), dtype=np.uint8)
        pts = deque(maxlen=512)

    if cv.waitKey(1) & 0xFF == ord('d'): 
        print("key d is pressed")
        notepad = np.zeros((490,640,3), dtype=np.uint8)
        pts = deque(maxlen=512)
            
    if cv.waitKey(1) & 0xFF == ord('a'):
        print("key a is pressed")
        if len(pts) != []:
            gray = cv.cvtColor(notepad, cv.COLOR_BGR2GRAY)
            contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            if len(contours) > 0:
                cnt = sorted(contours, key = cv.contourArea, reverse = True)[0]
                print(cv.contourArea(cnt))
                if cv.contourArea(cnt) > 1000: 
                    x, y, w, h = cv.boundingRect(cnt)
                    digit = gray[y-30:y + h + 30, x-30:x + w+30]
                    X = cv.resize(digit, (28, 28))
                    X = X.T
                    # X = cv.rotate(X, cv.ROTATE_90_COUNTERCLOCKWISE)
                    # X = cv.flip(X, 0)
                    print(X.shape)
                    result = model.predict(X)
                    print(result)
        notepad = np.zeros((490,640,3), dtype=np.uint8)
        pts = deque(maxlen=512)

    cv.putText(notepad, f"Prediction: {result}", (10, 470), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv.imshow('Video', img)
    # cv.imshow('mask',mask)
    # cv.imshow('res',res)
    cv.imshow('Note', notepad)
    
    k = cv.waitKey(1) & 0xFF
    if k == 27: # type esc to exit
        break


cap.release()
cv.destroyAllWindows()