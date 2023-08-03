import sys
import os
import threading

# Add the directory containing "simple" to Python path
simple_folder = os.path.join(os.path.dirname(__file__), "simple")
sys.path.append(simple_folder)

import tkinter as tk
import tkinter.messagebox
import customtkinter
import cv2
from PIL import Image, ImageTk
from keyPointstoNP import KeypointsProcessor
from simple.test import XYZProcessor
import queue

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App:
    def __init__(self, window, window_title):
        super().__init__()

        # # configure window
        self.window = window
        self.window.title(window_title)
        self.window.geometry(f"{1200}x{700}")

        self.window.grid_columnconfigure(1, weight=1)
        self.window.grid_columnconfigure((2, 3), weight=0)
        self.window.grid_rowconfigure((0, 1, 2), weight=1)
        self.kpProcesser = None
        self.predictions = ''

        # Create a queue to store frames
        self.frame_queue = queue.Queue()


        self.video_stream = cv2.VideoCapture(0)
        self.is_webcam_on = False
        self.is_signing = False
        self.FRAME_COUNT = 0

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self.window, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Sign Recognition", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=self.start_webcam, text='Start Webcam',)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=self.stop_webcam, text='Stop Webcam')
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(20, 20))

        custom_font =('Algerian',30)
        self.entry = customtkinter.CTkEntry(self.window, placeholder_text="Start signing", font=custom_font)
        self.entry.grid(row=4, column=1,columnspan=1,rowspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.main_button_1 = customtkinter.CTkButton(master=self.window, fg_color="transparent", border_width=2,height=50, text_color=("gray10", "#DCE4EE"), text='Start Sign', hover_color="#3669A0", state='disabled', command=self.start_signin)
        self.main_button_1.grid(row=4, column=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.canvas = tk.Canvas(self.window, width=1040, height=640)

        self.canvas.grid(row=0, column=1 ,columnspan=2 , padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.bar = customtkinter.CTkProgressBar(self.window,
                                  orientation='horizontal',
                                  mode='determinate')
    
        self.bar.grid(row=3, column=1, pady=10, padx=(20, 0), sticky="nsew")
    
        # Set default starting point to 0
        self.bar.set(0)

        self.update()

        self.appearance_mode_optionemenu.set("Dark")

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def start_webcam(self):
        self.is_webcam_on = True
        self.sidebar_button_1.configure(state='disabled')
        self.sidebar_button_2.configure(fg_color='red')
        self.main_button_1.configure(state='normal')

    
    def stop_webcam(self):
        self.is_webcam_on = False
        self.is_signing = False
        self.sidebar_button_1.configure(state='normal')
        self.sidebar_button_2.configure(fg_color="#3669A0")
        self.main_button_1.configure(state='disabled')
    
    def stop_siging(self):
        self.main_button_1.configure(fg_color="transparent", text='Start Sign', command=self.start_signin)
        self.is_signing = False

    def start_signin(self):
        self.main_button_1.configure(fg_color='green', text='Stop Signing', command=self.stop_siging)
        self.kpProcesser = KeypointsProcessor()
        self.FRAME_COUNT = 0;
        self.is_signing = True
        self.window.after(5000, self.stop_sign_after_time)

    def stop_sign_after_time(self):
        if self.is_signing:
            self.main_button_1.configure(fg_color='red', text='Loading')
            self.bar.start()
            print('FPS', self.video_stream.get(5))
            self.kpSize = self.kpProcesser.retriveFrameQue(self.frame_queue)
            self.frame_queue = queue.Queue()
            if self.kpSize:
                self.stop_recording()
            elif self.kpSize == 0:
                self.start_signin()

    def stop_recording(self):

        def process_xyz_array_thread():
            npXYZArray = self.kpProcesser.get_xyz_array()
            xyz_processor = XYZProcessor(npXYZArray)
            xyz_processor.process_xyz_array()
            self.predictions = xyz_processor.get_prediction()
            self.bar.stop()
            self.bar.set(1)
            if self.kpSize:
                self.entry.insert(tk.END,self.predictions)
            self.start_signin()

        # Start the thread
        processing_thread = threading.Thread(target=process_xyz_array_thread)
        processing_thread.start()

    def update(self):
        if self.is_webcam_on:
            ret, frame = self.video_stream.read()
            if self.is_signing:
                self.frame_queue.put(frame)
                self.FRAME_COUNT = self.FRAME_COUNT+1

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                width, height = img.size
                new_width = self.canvas.winfo_width()
                new_height = int(height * new_width / width)

                img = img.resize((new_width, new_height))
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.imgtk = imgtk
                self.canvas.create_image(0, 0, anchor=tkinter.NW, image=imgtk)
            

        self.window.after(1, self.update)


if __name__ == "__main__":
    root = customtkinter.CTk()
    app = App(root, "Sinhala Sign Recognition App")
    root.mainloop()