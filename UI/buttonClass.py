
import customtkinter
# First, let's define a base class for our buttons:
class BaseButton:
    def __init__(self, master, **kwargs):
        self.button = customtkinter.CTkButton(master, **kwargs)

    def grid(self, **kwargs):
        self.button.grid(**kwargs)
        
    def configure(self, **kwargs):
        self.button.configure(**kwargs)  # Configure the underlying widget

class StartWebcamButton(BaseButton):
    def __init__(self, master, command, **kwargs):
        super().__init__(master, command=command, text='Start Webcam', **kwargs)

class StopWebcamButton(BaseButton):
    def __init__(self, master, command, **kwargs):
        super().__init__(master, command=command, text='Stop Webcam', **kwargs)

class StartSignButton(BaseButton):
    def __init__(self, master, command, **kwargs):
        super().__init__(master, fg_color="transparent", border_width=2, height=50, text_color=("gray10", "#DCE4EE"), text='Start Sign', hover_color="#3669A0", state='disabled', command=command, **kwargs)
