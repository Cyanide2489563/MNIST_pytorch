import tkinter as tk
import PIL
import cv2
import numpy
import torch
from PIL import ImageDraw, Image
from torchvision.transforms import transforms

from MLP import MLP

width = 500
height = 500

white = (255, 255, 255)
green = (0, 128, 0)

window = tk.Tk()
window.geometry(str(width) + 'x' + str(height))
window.resizable(False, False)
window.title("QMNIST 手寫數字辨識")
top_frame = tk.Frame(window)


def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=40)
    draw.line([x1, y1, x2, y2], fill="black", width=40)


def process_image(pil_image):
    preprocess = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # preprocess the image
    img_tensor = preprocess(pil_image)

    # add dimension for batch
    img_tensor.unsqueeze_(0)

    return img_tensor


def identify():
    filename = "image.png"
    image.save(filename)

    img = cv2.imread(filename, 0)
    img = cv2.bitwise_not(img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = process_image(img)

    model = MLP()
    model.eval()
    model.load_state_dict(torch.load("model.pt"))

    output = model(img)
    pred = output.argmax(dim=1)
    print("Prediction: {}".format(pred[0].item()))


def clear():
    cv.delete('all')
    draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))


clear_button = tk.Button(text="清除", command=clear)
clear_button.pack()
identify_button = tk.Button(text="識別", command=identify)
identify_button.pack()

image = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image)

# 繪圖板
cv = tk.Canvas(window, width=width, height=height, bg='white')
cv.pack()

cv.pack(expand=tk.YES, fill=tk.BOTH)
# 綁定事件
cv.bind("<B1-Motion>", paint)

if __name__ == '__main__':
    window.mainloop()
