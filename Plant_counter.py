# -*- coding: utf-8 -*-


import os
import sys
import json
import time
import numpy as np
import cv2
from global_functions import ensureDir


TB_WIN_LABEL = "Trackbars"

MIN_HUE_TB_LABEL = "min Hue"
MIN_SAT_TB_LABEL = "min Sat"
MIN_VAL_TB_LABEL = "min Val"

MAX_HUE_TB_LABEL = "max Hue"
MAX_SAT_TB_LABEL = "max Sat"
MAX_VAL_TB_LABEL = "max Val"

MIN_HUE_VALUE = 0
MIN_SAT_VALUE = 0
MIN_VAL_VALUE = 0

MAX_HUE_VALUE = 180
MAX_SAT_VALUE = 255
MAX_VAL_VALUE = 255

# declaring min hue value for green
MIN_GREEN_HUE = 45
MAX_GREEN_HUE = 77
MIN_GREEN_SAT = 19
MAX_GREEN_SAT = 255
MIN_GREEN_VAL = 164
MAX_GREEN_VAL = 255

KERNEL_SIZE_TB_LABEL = "kernel size"
DEFAULT_KERNEL_SIZE = 2
MAX_KERNEL_SIZE = 10

ERODE_ITERATIONS_TB_LABEL = "erode"
DEFAULT_ERODE_ITERATIONS = 2
MAX_ERODE_ITERATIONS = 10

DILATE_ITERATIONS_TB_LABEL = "dilate"
DEFAULT_DILATE_ITERATIONS = 4
MAX_DILATE_ITERATIONS = 30

ESCAPE_KEY = 27
RESIZABLE_WINDOW = 0

def nothing(x):
  pass


def createTrackbars():
  cv2.namedWindow(TB_WIN_LABEL, RESIZABLE_WINDOW)
  cv2.createTrackbar(MIN_HUE_TB_LABEL, TB_WIN_LABEL, MIN_GREEN_HUE, MAX_HUE_VALUE, nothing)
  cv2.createTrackbar(MIN_SAT_TB_LABEL, TB_WIN_LABEL, MIN_GREEN_SAT, MAX_SAT_VALUE, nothing)
  cv2.createTrackbar(MIN_VAL_TB_LABEL, TB_WIN_LABEL, MIN_GREEN_VAL, MAX_VAL_VALUE, nothing)
  cv2.createTrackbar(MAX_HUE_TB_LABEL, TB_WIN_LABEL, MAX_GREEN_HUE, MAX_HUE_VALUE, nothing)
  cv2.createTrackbar(MAX_SAT_TB_LABEL, TB_WIN_LABEL, MAX_GREEN_SAT, MAX_SAT_VALUE, nothing)
  cv2.createTrackbar(MAX_VAL_TB_LABEL, TB_WIN_LABEL, MAX_GREEN_VAL, MAX_VAL_VALUE, nothing)
  cv2.createTrackbar(KERNEL_SIZE_TB_LABEL, TB_WIN_LABEL, DEFAULT_KERNEL_SIZE, MAX_KERNEL_SIZE, nothing)
  cv2.createTrackbar(ERODE_ITERATIONS_TB_LABEL, TB_WIN_LABEL, DEFAULT_ERODE_ITERATIONS, MAX_ERODE_ITERATIONS, nothing)
  cv2.createTrackbar(DILATE_ITERATIONS_TB_LABEL, TB_WIN_LABEL, DEFAULT_DILATE_ITERATIONS, MAX_DILATE_ITERATIONS, nothing)


def updateJson(srcPath, dstPath, data):
  srcDirname, srcImgname = os.path.split(srcPath)
  dstDirname, dstImgname = os.path.split(dstPath)
  jsonPath = os.path.join(dstDirname, "details.json")
  
  jsonData = {}
  jsonData["source"] = srcImgname
  jsonData["images"] = {}
  
  if os.path.exists(jsonPath):
    infile = open(jsonPath, "r")
    jsonData = json.loads(infile.read())
    infile.close()
  
  operations = []
  operations.append("h " + data["minH"] + "-" + data["maxH"])
  operations.append("s " + data["minS"] + "-" + data["maxS"])
  operations.append("v " + data["minV"] + "-" + data["maxV"])
  operations.append("erode " + data["erodeKernel"] + " x" + data["erodeIterations"])
  operations.append("dilate " + data["dilateKernel"] + " x" + data["dilateIterations"])
    
  jsonData["images"][dstImgname] = {}
  jsonData["images"][dstImgname]["date"] = time.strftime("%Y-%m-%d", time.gmtime())
  jsonData["images"][dstImgname]["operations"] = operations
  
  with open(jsonPath, 'w') as outfile:
    json.dump(jsonData, outfile, indent = 2)


def main(srcPath, dstPath):
  SMALL_FACTOR = 0.3 
  
  src = cv2.imread(srcPath)     # read the source file
  small = cv2.resize(src, (src.shape[1],src.shape[0]), fx=SMALL_FACTOR, fy=SMALL_FACTOR)      # ressize the image  to new dim
  # small = cv2.resize(src, (0, 0), fx=SMALL_FACTOR, fy=SMALL_FACTOR)
  gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)                        # convert to gray scale
  hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)                          # convert to HSV
  
  if dstPath != None:
    ensureDir(dstPath)
  
  # createTrackbars()
  # cv2.namedWindow("Source", RESIZABLE_WINDOW)
  # cv2.imshow("Source", src)
  # print("Press Escape to quit.")
  
        
  minHue = cv2.getTrackbarPos(MIN_HUE_TB_LABEL, TB_WIN_LABEL)
  minSat = cv2.getTrackbarPos(MIN_SAT_TB_LABEL, TB_WIN_LABEL)
  minVal = cv2.getTrackbarPos(MIN_VAL_TB_LABEL, TB_WIN_LABEL)
    
  maxHue = cv2.getTrackbarPos(MAX_HUE_TB_LABEL, TB_WIN_LABEL)
  maxSat = cv2.getTrackbarPos(MAX_SAT_TB_LABEL, TB_WIN_LABEL)
  maxVal = cv2.getTrackbarPos(MAX_VAL_TB_LABEL, TB_WIN_LABEL)
    
  kernelSize = 2
  dilateIterations = 4
  erodeIterations = 2
    
  lower = np.array([45, 19, 169])
  upper = np.array([77, 255, 255])
  mask = cv2.inRange(hsv, lower, upper)
  kernel = np.ones((kernelSize, kernelSize), np.uint8)

    
  masked = cv2.bitwise_and(gray, gray, mask=mask)
  masked = cv2.threshold(masked, 5, 255, cv2.THRESH_BINARY)[1]
  masked = cv2.erode(masked, kernel, iterations = erodeIterations)
  masked = cv2.dilate(masked, kernel, iterations = dilateIterations)
  cv2.imshow('masked image',masked)
  cv2.waitKey()
  cv2.destroyAllWindows()
  contours, _ = cv2.findContours(masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contoursLen = len(contours)
  plantsNumber = 0
  colorStep = int(200.0/contoursLen)
  PERIMETER_LIMIT = 20
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
    
  # if dstPath != None:
    # data = {}
    # data["minH"] = str(minHue)
    # data["maxH"] = str(maxHue)
    # data["minS"] = str(minSat)
    # data["maxS"] = str(maxSat)
    # data["minV"] = str(minVal)
    # data["maxV"] = str(maxVal)
    # data["erodeKernel"] = "(" + str(kernelSize) + "," + str(kernelSize) + ")"
    # data["erodeIterations"] = str(erodeIterations)
    # data["dilateKernel"] = "(" + str(kernelSize) + "," + str(kernelSize) + ")"
    # data["dilateIterations"] = str(dilateIterations)
    
    # updateJson(srcPath, dstPath, data)
    # cv2.imshow('masked',masked)  
    # cv2.imwrite(dstPath, masked)
    
  cv2.destroyAllWindows()


def printUsage():
  print("""
  USAGE:
  python Plant_counter.py --src <img-path> [--dst <img-path>]
  e.g.: 
  python Plant_counter.py --src foo/bar.jpg
  """)

def parseArgs(args):
  src, dst = None, None
  
  for i in range(len(args)):
    try:
      if args[i] == "--src":
        src = args[i+1]
      elif args[i] == "--dst":
        dst = args[i+1]
    except:
      break
  
  if src == None:
    printUsage()
    sys.exit()
    
  return src, dst

  
if __name__ == "__main__":
  if len(sys.argv) > 1:
    src, dst = parseArgs(sys.argv[1:])
    main(src, dst)
  else:
    printUsage()
