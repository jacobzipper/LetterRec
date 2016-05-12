import cv2
import numpy as np
import math
def checkRowHasBlack(row):
    hasBlack = False
    for a in row:
        if np.sum(a) < 50:
            hasBlack = True
            break
    return hasBlack
def scanCrop(img):
    firstBlack = False
    indexes = [-1,-1]
    counter = 0
    for a in img:
        for b in a:
            if np.sum(b) < 50 and firstBlack==False:
                indexes[0] = counter
                firstBlack = True
        if firstBlack and checkRowHasBlack(a) == False:
            indexes[1] = counter
        if indexes[0] != -1 and indexes[1] != -1:
            break
        counter+=1
    return indexes
def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    
def nextLetter(thresh1):
    thresh1 = thresh1[::-1]
    indexes=[0,1]
    indexes = scanCrop(thresh1)
    print indexes
    thresh1 = thresh1[indexes[0]:indexes[1]]
    thresh1 = rotate_about_center(thresh1,90)
    thresh1 = thresh1[10:,:len(thresh1[0])-3]
    indexes = scanCrop(thresh1)
    print indexes
    thresh1 = thresh1[indexes[0]:indexes[1]]
    return thresh1

filename = 'Temp1.jpg'
img = cv2.imread(filename)
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
res = cv2.resize(img,None,fx=.25, fy=.25, interpolation = cv2.INTER_CUBIC)
ret,thresh1 = cv2.threshold(res,50,255,cv2.THRESH_BINARY)
thresh1 = nextLetter(thresh1)

# show image
cv2.imshow('dst',thresh1)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()