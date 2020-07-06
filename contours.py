# -*- coding: utf-8 -*-
# Python 2
# OpenCV required

import os
import sys
import numpy as np
import cv2
from global_functions import ensureDir

def main(path):
  src = cv2.imread(path)
  src = cv2.resize(src, (0,0), fx=0.7, fy=0.7)
  gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 127, 255, 0)[1]
  
  _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contoursLen = len(contours)
  plantsNumber = 0
  colorStep = int(200.0/contoursLen)
  PERIMETER_LIMIT = 30
  LINE_WIDTH = 2
  
  for i in range(contoursLen):
    perimeter = cv2.arcLength(contours[i], True)
    if perimeter > PERIMETER_LIMIT:
      plantsNumber += 1
      val = (i+1) * colorStep
      cv2.drawContours(src, [contours[i]], -1, (val,val,val), LINE_WIDTH)
      print("(" + str(val) + "," + str(val) + "," + str(val) + ") : " + str(perimeter))
  
  print("\n" + str(plantsNumber) + " plants.")
  
  cv2.imshow("Contours", src)
  cv2.waitKey()
  cv2.destroyAllWindows()


def printUsage():
  print("""
  USAGE:
  python contours.py <img-path>
  e.g.: python contours.py bar/foo.jpg
  """)
  
    
if __name__ == "__main__":
  if len(sys.argv) > 1:
    src = sys.argv[1]
    main(src)
  else:
    printUsage()
