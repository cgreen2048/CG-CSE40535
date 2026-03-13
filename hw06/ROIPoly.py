import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image, ImageDraw, ImageOps
import math
import cv2

class roiPoly():
    def __init__(self, max_points = 4, delete_dist_limit=42, sort=False):
        self.points = []
        self.image = None
        self.delete_dist_limit = delete_dist_limit
        self.max_points = max_points
        self.origImage = None
        self.delete_bool = False
        self.sort = False

        self.root = Tk()
        self.root.title("Select 4 points for warping (clockwise starting from top left)")

        self.select_image_button = Button(self.root, text="Select Image", command=self.select_image)
        self.delete_button = Button(self.root, text="Delete All Points", state=DISABLED, relief=RAISED, command=self.delete_point)
        self.confirm_button = Button(self.root, text="Confirm Selection", state=DISABLED, relief=RAISED, command=self.confirm)
        self.select_image_button.grid(row=0, column=0)
        self.delete_button.grid(row=0, column=1)
        self.confirm_button.grid(row=0, column=2)
        self.canvas = Canvas(self.root)
        self.canvas.bind("<Button-1>", self.callback)
        self.canvas.grid(row=1, columnspan=10)
        self.root.mainloop()

    def make_tuple(self, points):
        return ((points[0][0], points[0][1]), (points[1][0], points[1][1]), (points[2][0], points[2][1]), (points[3][0], points[3][1]))

    def create_image_with_points(self):
        self.image = self.origImage.copy().convert('RGB')

        width, height = self.image.size
        draw = ImageDraw.Draw(self.image)

        for point in self.points:
            draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill='red')
        
        if len(self.points) == 4:
            points_t = self.make_tuple(self.points)
            draw.polygon((points_t), outline='red')
        
        self.image = ImageTk.PhotoImage(self.image)
        return self.image

    def callback(self, event):
        #print("clicked at", event.x, event.y)
        x = int(event.x) 
        y = int(event.y)
        if len(self.points) < 4:
            self.points.append([int(x), int(y)])
            if len(self.points) == 4:
                if self.sort:
                    self.setcenter()
                    self.points = sorted(self.points, key=self.angle)
                self.confirm_button['state']='normal'
        else:
            print('Cannot select more than 4 points.')

        if self.image is not None:
            self.image = self.create_image_with_points()    
            self.canvas.create_image(0, 0, anchor=NW, image=self.image)

    def select_image(self):
        filename = filedialog.askopenfilename()
        self.origImage = Image.open(filename).convert('RGB')
        self.points = []
        self.image = self.create_image_with_points()
        self.canvas.config(width = self.image.width(), height = self.image.height())
        self.canvas.create_image(0, 0, anchor=NW, image=self.image)
        self.delete_button['state']='normal'

    def delete_point(self):
        self.points = []
        self.confirm_button['state']='disabled'
        self.image = self.create_image_with_points()
        self.canvas.create_image(0, 0, anchor=NW, image=self.image)

    def confirm(self):
        self.root.destroy()

    def setcenter(self):
        self.centerx = np.mean([point[0] for point in self.points])
        self.centery = np.mean([point[1] for point in self.points])

    def angle(self, point):
        x = point[0] - self.centerx
        y = point[1] - self.centery
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return phi


if __name__ == "__main__":

    gui = roiPoly()
    print(gui.points)

    print("Using CV2 perspective transform to get the correct image")

    src = np.array(gui.points).astype(np.float32)
    image_np = cv2.cvtColor(np.array(gui.origImage), cv2.COLOR_RGB2BGR)

    w, h = gui.image.width(), gui.image.height()
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]).astype(np.float32)

    H_mat = cv2.getPerspectiveTransform(src, dst)

    out = cv2.warpPerspective(image_np, H_mat, (w, h), flags=cv2.INTER_LINEAR)

    cv2.imshow('Warped image', out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
