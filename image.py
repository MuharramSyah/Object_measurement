import imutils
import time
import cv2
import numpy as np

print("[INFO] detecting 'DICT_5x5_1000' tags...")
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

if __name__ == "__main__":
    print("[INFO] Running program...")

    img = cv2.imread('sample.png')
    img = imutils.resize(img, width=1080)

    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            # get width and height marker
            int_corners = np.int0(markerCorner)
            width = np.linalg.norm(topLeft - topRight)
            height = np.linalg.norm(topRight - bottomRight)

            # get parimeter marker
            aruco_parimeter = cv2.arcLength(markerCorner[0], True)
            pixel_to_cm = aruco_parimeter / 20
            print(f'1 cm : {pixel_to_cm}, width: {width}, height: {height}')

            # get real size object width and height
            object_width = width / pixel_to_cm
            object_height = height / pixel_to_cm

            print(f'Object Width : {object_width}, Object Height : {object_height}')

            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(img, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(img, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(img, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
            
            print("[INFO] ArUco marker ID: {}".format(markerID))
            # show the output image
            cv2.imshow("Frame",img)    
            cv2.waitKey(0)
