#importing all the required package
from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
import time
import os
#image name here image and py file are in same folder
image = filedialog.askopenfilename(initialdir="C:/",title="Please Select a image")
img_obj = cv2.imread(image)
print(img_obj.shape)   #-- to check the shape of image 
scale_percent = 0.50
width = int(img_obj.shape[1]*scale_percent)
height = int(img_obj.shape[0]*scale_percent)
dim = (width,height)
resized = cv2.resize(img_obj,dim,interpolation = cv2.INTER_AREA)  # resizing the image 
cv2.imwrite("resize.jpg",resized)
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])
sharpened = cv2.filter2D(resized,-1,kernel_sharpening)   # shape the image
gray = cv2.cvtColor(sharpened , cv2.COLOR_BGR2GRAY)      # convert in black and white 
objectDetection = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV )  #convert in image detection formate
inv = 255-gray					# convert in inverse form 
gauss = cv2.GaussianBlur(inv,ksize=(15,15),sigmaX=0,sigmaY=0)  # convert in gauss form 
pencil = cv2.divide(gray,255-gauss,scale=256)

# to display these four images 
# cv2.imshow('resized',resized)
# cv2.imshow('sharp',sharpened)
# cv2.imshow("gray", gray)
# cv2.imshow('pencile',pencil)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#Here u can save the image , that formate you want.Here, image will save at the same path of code file 
cv2.imwrite("pencilSketch.jpg",pencil)
tk = Tk()
canvas = Canvas(tk, width = 1500,height = 800)
tk.title('Sketch')
canvas.pack()
resolution = 1
img = cv2.imread('pencilSketch.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img1 = cv2.imread('resize.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
width = len(img[0])
height = len(img)
class Pix():
    def __init__(self,canvas,x,y,c,w,h):
        self.canvas = canvas
        self.x = x
        self.y = y
        hexa = '#{:02x}{:02x}{:02x}'.format(c[0],c[1],c[2])
        self.body = self.canvas.create_rectangle(25+x-w/2, 75+y-h/2, 25+x+w/2, 75+y+h/2,fill = hexa,outline = '')
class Pixel():
    def __init__(self,canvas,x,y,c,w,h):
        self.canvas = canvas
        self.x = x
        self.y = y
        hexa = '#{:02x}{:02x}{:02x}'.format(c[0],c[1],c[2])
        self.body = self.canvas.create_rectangle(700+x-w/2, 75+y-h/2, 700+x+w/2, 75+y+h/2,fill = hexa,outline = '')
tk.update()
w = canvas.winfo_width()/(2*width*resolution)
h = canvas.winfo_height()/(2*height*resolution)
yy = -1
for y in range(0,height,round(1/resolution)):
    yy +=1
    xx = 0
    for x in range(0,width,round(1/resolution)):
        xx += 1
        r = Pix(canvas, xx, yy, img1[y][x], w, h)        
pixels = []
tk.update()
w = canvas.winfo_width()/(2*width*resolution)
h = canvas.winfo_height()/(2*height*resolution)
yy = -1
for y in range(0,height,round(1/resolution)):
    yy +=1
    xx = 0
    for x in range(0,width,round(1/resolution)):
        xx += 1
        p = Pixel(canvas, xx, yy, img[y][x], w, h)
    pixels.append(p)
    tk.update()
while True:
    try:
        tk.update()
        time.sleep(0.09)
    except:
        break
os.remove("resize.jpg")