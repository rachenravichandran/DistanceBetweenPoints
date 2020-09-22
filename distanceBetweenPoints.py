from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import serial

capDeviceNum = 0
actualWidth = 20 # width of Standard square
bias = 0 # offset value for any error in distance measurement
distance = 0 # actual distance measured
resizeRatio = 1.1 # ratio for resizing the captured camera frame

blurSize = 15 # blurring Gaussian kernel size
dilateIter = 5 # no. of iterations for dilation
erodeIter = 3 # no. of iterations for erosion
cannyUpperThreshold = 100 # Upper hysteresis threshold for Canny
cannyLowerThreshold = 50 # Lower hysteresis threshold for Canny

blackThreshold = 30 # Threshold value for black
# Below arrays represent HSV values. 0-180 for Hue, 0-255 for Saturation and 0-255 for Value
lowerRed = np.array([0,150,30]) # Lower range for detection of red
upperRed = np.array([7,255,255]) # Upper range for detection of red

cap= cv2.VideoCapture(capDeviceNum) #capture video from device 1 (device 0 is integrated webcam, 1 is external USB camera ...)
    
def findMidPoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def findMidTopBotCenter(box):
    (tl, tr, br, bl) = box # extracts top left, top right, bottom right, bottom left
    midTop = findMidPoint(tl, tr)
    midBot = findMidPoint(bl, br)
    center = findMidPoint(midTop,midBot) # center of box
    midLeft = findMidPoint(tl, bl)
    midRight = findMidPoint(tr, br)
    sortedBox = list(box)
    sortedBox.sort(key = lambda x: x[1])
    lowest = findMidPoint( sortedBox[2],sortedBox[3])
    highest = findMidPoint(sortedBox[0],sortedBox[1])
    return (midTop,midBot,center,midLeft,midRight,lowest,highest)

def grabContoursFromChannel(image,i=0):
    
    image = cv2.dilate(image, None, iterations=dilateIter) # dilate image for 'dilateIter' times
    image = cv2.erode(image, None, iterations=erodeIter) # erode image for 'erodeIter' times
    
    # canny edge the image for edge detection so that only continuous boundaries are selected
    edged = cv2.Canny(image, cannyLowerThreshold, cannyUpperThreshold) 
    
    edged = cv2.dilate(edged, None, iterations=dilateIter) # dilate image for 'dilateIter' times
    edged = cv2.erode(edged, None, iterations=erodeIter) # erode image for 'erodeIter' times
    
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finds all the contours in the copy of edged image
    if cnts[0] == []:
        return False
    cnts = imutils.grab_contours(cnts) #   removes unnecessary values from cnts and returns only the coordinates in a tuple format
    (cnts, _) = contours.sort_contours(cnts) # sorts all contours from top to bottom, left to right
    count = 0
    for cnts_i in cnts:
        # returns the first contour with area greater than 100
        if cv2.contourArea(cnts_i) > 100:
            if count == i:
                box = cv2.minAreaRect(cnts_i) # draw a bounding box with minimum area around the contour
                #convert the rectangular coordinates in BoxPoints() format to array format (x1,y1),...,(x4,y4) 
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                #Order the rectangular coordinates of a box in clockwise direction
                box = perspective.order_points(box)
                return box
            count += 1
    # if no contour is detected, it returns false
    return False
    
def separateImageChannels(originalImg):
    
    #Separate Red channel in image
    redImg = cv2.cvtColor(originalImg,cv2.COLOR_BGR2HSV)
    redImg = cv2.inRange(redImg, lowerRed, upperRed)
    
    #Threshold Black image
    blackImage = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)
    blurredBlackImage = cv2.GaussianBlur(blackImage, (blurSize, blurSize), 0)
    _,blackImg = cv2.threshold(blurredBlackImage,blackThreshold,255,cv2.THRESH_BINARY_INV)
    
    # Grab contour boxes for each image channel and store it as an array
    contoursArr = []
    images = [blackImg,blackImg,redImg]
    for i,image in enumerate(images):
        if i == 1:
            temp = grabContoursFromChannel(image,i)
        else:
            temp = grabContoursFromChannel(image)
        if np.any(temp) == False:
            return False
        else:
            contoursArr.append(temp)
    return contoursArr

def lineCheck(lineShow):
    # lineShow[0][1] is Black's top row and lineShow[1][1] is Red's bottom row
    # Checks if black top is lesser than red bottom (i.e. black top is located above (then value is less))
    if(lineShow[0][1]<=lineShow[1][1]):
        # assign black top coordinate  to red bottom
        lineShow[1] = lineShow[0]
    return lineShow

while(cap.isOpened()):
    
    # Read the frame from camera
    ret, frame = cap.read()
    if ret == False:
        print("Device can't be opened")
        break
    
    # resizes the captured frame by a factor of 'resizeRatio' to match the real world dimensions to monitor screen
    # e.g. 2 mm in reality is shown as 2 mm on laptop screen too. This ratio changes with monitor
    image = cv2.resize(frame,(int(frame.shape[1]*resizeRatio),int(frame.shape[0]*resizeRatio)))
    
    img = image.copy() # all the contours, distance and lines will be drawn on this image
    
    contoursArr = separateImageChannels(image.copy()) # holds the contours [black,red,blue] as rectangles
    # If any one of the contour is not detected
    if contoursArr != False:
        midPoints = [] # holds the mid points of contours
        lineShow = [] # holds the coordinates for line segment between two fingers
        pixelsPerMetric = None # pixel ratio for conversion of pixels to mm (unit pixels/mm)
        
        i = 0 # contour count
        for box in contoursArr:
            i=i+1
            # draws the contour in original image (just for display purpose)
            cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
            
            (midTop,midBot,center,midLeft,midRight,lowest,highest) = findMidTopBotCenter(box)

            # in first contour (standard box), pixel ratio is calculated as (actual distance in pixels/actual distance in mm)
            if pixelsPerMetric is None:
                dA = dist.euclidean(midLeft, midRight)
                pixelsPerMetric = dA / actualWidth
                
            if i == 1:
                continue
            if i == 3:
                lineShow.append(lowest) # Append the midBottom point of box if red contour
            if i == 2:
                lineShow.append(highest)# Append the midTop point of box if blue contour
                
            midPoints.append(center) # append the midpoints of contour
        
        lineShow = lineCheck(lineShow) # check if the line coordinates midTop of red is to the top of midBot of blue
        
        # once all the 3 contours are grabbed:
        # find the euclidean distance between the two coordinates of black spots (midBot and midTop)
        distance = dist.euclidean(lineShow[0], lineShow[1])
        
        # Calculates the actual distance. Bias is used to offset any erroneous measurement
        distance = distance / pixelsPerMetric
        
        # draw the line segments
        cv2.line(img, (int(lineShow[0][0]),int(lineShow[0][1])), (int(lineShow[1][0]),int(lineShow[1][1])),(0, 255, 0), 2)
        
        # find the midpoint of line segment and label the distance just over the line
        midLine = findMidPoint(midPoints[0],midPoints[1])
        cv2.putText(img, "{:.1f}mm".format(distance), (int(midLine[0]),int(midLine[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 1)
    
    cv2.imshow("Contours Image", img) # image used for reference purpose which shows the contour, distance and line
    
    # if spacekey is pressed, breaks the loop
    keyPress = cv2.waitKey(5)&0xFF
    if keyPress == 32:
        break 

# closes all the windows and releases the camera
cap.release()
cv2.destroyAllWindows()