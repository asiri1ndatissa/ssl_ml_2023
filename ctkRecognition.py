import tkinter as tk
import customtkinter
from ctkApp import App


class MainApp(tk.Tk):  # Inherits from tk.Tk
    def __init__(self):
        super().__init__()
        self.title("Main App")
        self.geometry(f"{1100}x{1000}")

        # Instantiate the App class and pass self as the parent window
        self.app = App(self, "Webcam App")

if __name__ == "__main__":
    customtkinter.set_appearance_mode("System")
    customtkinter.set_default_color_theme("blue")

    root = MainApp()
    root.mainloop()
