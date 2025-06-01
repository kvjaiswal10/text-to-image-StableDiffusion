import tkinter as tk
import customtkinter as ctk

from PIL import ImageTk # display generated img in gui
from authtoken import auth_token # huggingface token

import torch # to run on gpu
from torch import autocast

from diffusers import StableDiffusionPipeline


# create the app
app = tk.Tk()
app.geometry("532x632")
app.title("StableDiffusion application")
ctk.set_appearance_mode("dark")

# input field
prompt = ctk.CTkEntry(app, height=40, width=512, font=("Arial", 20), 
                     text_color="black", fg_color="white")
prompt.place(x=10, y=10)

# image display placeholder
lmain = ctk.CTkLabel(app, height=512, width=512)
lmain.place(x=10, y=110)

# load stable diffusino model using token
modelid = "CompVis/stable-diffusion-v1-4"
# move model to gpu memory
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtypes=torch.float16,
                                               use_auth_token=auth_token)
pipe.to(device)

"""modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"  # Use CPU instead of CUDA

pipe = StableDiffusionPipeline.from_pretrained(
    modelid,
    revision="fp16",  # You may also need to remove this
    torch_dtype=torch.float16,  # float16 is not fully supported on CPU
    use_auth_token=auth_token
)
"""
pipe.to(device)


def generate():
    with autocast(device):
        # generate image based on prompt
        # get first image from pipeline input
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]
    
    image.save("generatedimage.png")
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)

# generate button
trigger = ctk.CTkButton(app,height=40, width=120, font=("Arial", 20), 
                     text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)


app.mainloop()
