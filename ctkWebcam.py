import customtkinter as ctk


import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import subprocess

import time
import torch
import torch.nn as nn

class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry(f"{1100}x{1000}")
         # configure grid layout (4x4)
        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_columnconfigure((2, 3), weight=0)
        self.window.grid_rowconfigure((0, 1, 2), weight=1)

        self.entry = ctk.CTkEntry(self.window, placeholder_text="CTkEntry")
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.grid(row=0, column=0, columnspan=3)
        self.video_stream = cv2.VideoCapture(0)
        self.update()
        # self.entry.pack()

    def update(self):
        ret, frame = self.video_stream.read()
        if ret:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.imgtk = imgtk
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
 
        self.window.after(2, self.update)


# window = ctk.CTk()
# window.title("Sign Recognition")

# window.geometry(f"{1100}x{500}")
# label = ctk.CTkLabel(window, text="Sign Recognition", font=ctk.CTkFont(size=20, weight="bold"))

# label.pack()
# window.mainloop()

if __name__ == "__main__":
    root = ctk.CTk()
    app = WebcamApp(root, "Webcam App")
    root.mainloop()

