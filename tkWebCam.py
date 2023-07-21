import sys
import os
import threading


# Add the directory containing "simple" to Python path
simple_folder = os.path.join(os.path.dirname(__file__), "simple")
sys.path.append(simple_folder)


import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import subprocess

import time
import torch
import torch.nn as nn
from keyPointstoNP import KeypointsProcessor
from simple.test import XYZProcessor,callback_function

class FeatureGen(nn.Module):
    def __init__(self):
        super(FeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        LEN = x.shape[0]

        return x.reshape(LEN, -1)
    
feature_converter = FeatureGen()
class WebcamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_stream = cv2.VideoCapture(0)

        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.grid(row=0, column=0, columnspan=3)

        self.btn_start_webcam = tk.Button(window, text="Start Webcam", width=20, command=self.start_webcam)
        self.btn_start_webcam.grid(row=1, column=0, pady=10)

        self.btn_stop_webcam = tk.Button(window, text="Stop Webcam", width=20, command=self.stop_webcam, state=tk.DISABLED)
        self.btn_stop_webcam.grid(row=2, column=0, pady=5)

        self.btn_start_recording = tk.Button(window, text="Start Recording", width=20, command=self.start_recording, state=tk.DISABLED)
        self.btn_start_recording.grid(row=3, column=0, pady=10)

        self.btn_stop_recording = tk.Button(window, text="Stop Recording", width=20, command=self.stop_recording, state=tk.DISABLED)
        self.btn_stop_recording.grid(row=4, column=0, pady=5)

        self.text_area = tk.Text(window, width=30, height=10, font=("Arial", 24))  # Set font size to 12
        self.text_area.grid(row=1, column=1, rowspan=4, padx=10)

        self.is_webcam_on = False
        self.is_recording = False
        self.video_writer = None
        self.FRAME_COUNT = 0
        self.xyz_array = np.empty((0, 3))  # Initialize an empty array to store XYZ data
        self.predictions = ''
        self.kpProcesser = None

        self.update()

        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def start_webcam(self):
        self.is_webcam_on = True
        self.btn_start_webcam.config(state=tk.DISABLED)
        self.btn_stop_webcam.config(state=tk.NORMAL)
        self.btn_start_recording.config(state=tk.NORMAL)
        # self.get_keypoints()
        # self.window.after(5000, self.stop_webcam)

    def stop_webcam(self):
        

        self.is_webcam_on = False
        self.btn_start_webcam.config(state=tk.NORMAL)
        self.btn_stop_webcam.config(state=tk.DISABLED)
        self.btn_start_recording.config(state=tk.DISABLED)
        self.btn_stop_recording.config(state=tk.DISABLED)
        self.xyz_array = np.empty((0, 3))
        self.FRAME_COUNT = 0
        cv2.destroyAllWindows()


    def start_recording(self):
        self.is_recording = True
        self.btn_start_recording.config(state=tk.DISABLED)
        self.btn_stop_recording.config(state=tk.NORMAL)

        self.kpProcesser = KeypointsProcessor()

        self.window.after(5000, self.stop_recording_after_time)

    def stop_recording_after_time(self):

            if self.is_recording:
                self.stop_recording()


    def stop_recording(self):

        self.is_recording = False
        self.btn_start_recording.config(state=tk.NORMAL)
        self.btn_stop_recording.config(state=tk.DISABLED)

        def process_xyz_array_thread():
            npXYZArray = self.kpProcesser.get_xyz_array()
            xyz_processor = XYZProcessor(npXYZArray)
            xyz_processor.process_xyz_array()
            predicted_text = xyz_processor.get_prediction()
            self.predictions = xyz_processor.get_prediction()

            self.text_area.delete("1.0", tk.END)  # Clear existing text
            self.text_area.insert(tk.END, self.predictions)
            # Send the predicted text back to test.py using the callback function
            callback_function(predicted_text)

        # Start the thread
        processing_thread = threading.Thread(target=process_xyz_array_thread)
        processing_thread.start()

        # Print or use the predicted texts as needed
        print(self.predictions)


    def save_nparray(self):
        # Add the "scripts" folder to Python path
        scripts_folder = os.path.join(os.path.dirname(__file__), "simple")
        sys.path.append(scripts_folder)
        test_script = "test.py"
        command = ["python", os.path.join(scripts_folder, test_script), temp_file_path]
        subprocess.run(command)
        # os.remove(temp_file_path)

    def update(self):
        if self.is_webcam_on:
            ret, frame = self.video_stream.read()
            if self.is_recording:
                # self.video_writer.write(frame)
                self.kpProcesser.keyPointstoNP(frame)
            if ret:

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                width, height = img.size
                new_width = self.canvas.winfo_width()
                new_height = int(height * new_width / width)
                img = img.resize((new_width, new_height))
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.imgtk = imgtk
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        # Update the text area with the latest predicted texts
        self.text_area.delete("1.0", tk.END)  # Clear existing text
        self.text_area.insert(tk.END, self.predictions)
            

        self.window.after(2, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root, "Webcam App")
    root.mainloop()
