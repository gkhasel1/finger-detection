import cv2
import numpy
import math

video = cv2.VideoCapture(0)
cv2.namedWindow("Modified")
cv2.namedWindow("Original")

# Also just super janky
lower = numpy.array([0, 48, 80], dtype = "uint8")
upper = numpy.array([20, 255, 255], dtype = "uint8")

HSV_skin_array = []


def onClick(event, x, y, flags, image):
    print x,y
    if image: print "IMAGE"


while(video.isOpened()):
    # Get some video
    ret, image = video.read()

    # Get skin tone
    # while len(HSV_skin_array) < 10:
    #     cv2.setMouseCallback("Original", onClick)
    #     ret, image = video.read()
    #     cv2.imshow("Original", image)

    # convert it to the HSV color space, get mask using ranges
    frame = image.copy()
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)
    # bit wise and to extract image with mask
    skin = cv2.bitwise_and(image, image, mask = skinMask)
    gray_skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    ret, skin_thresh = cv2.threshold(gray_skin, 0, 255, cv2.THRESH_BINARY)

    # Finding largest contour
    contours, hierarchy = cv2.findContours(skin_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for i in range(len(contours)):
            largest_contour = contours[i]
            area = cv2.contourArea(largest_contour)
            if (area > max_area):
                max_area = area
                ci = i
    largest_contour = contours[ci]
    # Getting the convex hull and defects
    hull = cv2.convexHull(largest_contour)
    hull_no_points = cv2.convexHull(largest_contour, returnPoints=False)

    # Draw it on the screen
    drawing = numpy.zeros(image.shape, numpy.uint8)
    moments = cv2.moments(largest_contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00']) # cx = M10/M00
        cy = int(moments['m01'] / moments['m00']) # cy = M01/M00

    center = (cx,cy)
    cv2.circle(drawing, center, 5, [255,255,255], 2)
    cv2.drawContours(drawing, [largest_contour], 0, (255,255,0), 2)
    cv2.drawContours(drawing, [hull], 0, (0,255,0), 2)

    # Get defects
    defects = cv2.convexityDefects(largest_contour, hull_no_points)
    fingers = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i, 0]
        start = tuple(largest_contour[s][0])
        end = tuple(largest_contour[e][0])
        far = tuple(largest_contour[f][0])
        dist_center_hull = cv2.pointPolygonTest(hull, center, True)
        dist_far_hull = cv2.pointPolygonTest(hull, far, True)
        dist_far_center = math.hypot(center[0] - far[0], center[1] - far[1])
        far_inside_hull = cv2.pointPolygonTest(hull, far, False)
        # 20...what a janky ass piece of mutha fuckin shit.
        if dist_far_hull > 20:
            fingers += 1
        cv2.circle(drawing, far, 5, [0,0,255], -1)

    # -1 is also just so fuckin JANKY
    finger_string = "FINGERS :: {}".format(fingers - 1)
    # This is what the computer s
    cv2.imshow("Modified", drawing)
    # This is the actual image
    cv2.putText(image, finger_string, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), 4);
    cv2.imshow("Original", image)

    # wait 10 ms for input from user, if they hit esc quit
    k = cv2.waitKey(10)
    if k == 27: break

video.release()
cv2.destroyAllWindows()
