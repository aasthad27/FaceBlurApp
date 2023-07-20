import cv2
import tkinter as tk
from tkinter import filedialog
from mtcnn import MTCNN
import math
from PIL import Image, ImageTk
def detect_and_blur_faces(frame):
    detector = MTCNN()
    output = detector.detect_faces(frame)

    for single_output in output:
        x, y, width, height = single_output['box']
        face = frame[y:y+height, x:x+width]
        blurred_face = cv2.GaussianBlur(face, (55, 55), 30)
        frame[y:y+height, x:x+width] = blurred_face

    return frame
class FaceBlurApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Blur App")

        self.video_source = 0  # Default camera
        self.cap = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(root, width=self.cap.get(3), height=self.cap.get(4))
        self.canvas.pack()

        self.btn_open_file = tk.Button(root, text="Open Video File", command=self.open_file)
        self.btn_open_file.pack(pady=10)

        self.btn_start_blur = tk.Button(root, text="Start Blur", command=self.start_blur)
        self.btn_start_blur.pack(pady=5)

        self.btn_stop_blur = tk.Button(root, text="Stop Blur", command=self.stop_blur, state=tk.DISABLED)
        self.btn_stop_blur.pack(pady=5)

        self.is_blur_on = False
        self.update()

    def open_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.cap.release()
            self.cap = cv2.VideoCapture(file_path)

    def start_blur(self):
        self.is_blur_on = True
        self.btn_start_blur.config(state=tk.DISABLED)
        self.btn_stop_blur.config(state=tk.NORMAL)

    def stop_blur(self):
        self.is_blur_on = False
        self.btn_start_blur.config(state=tk.NORMAL)
        self.btn_stop_blur.config(state=tk.DISABLED)

    def update(self):
        ret, frame = self.cap.read()

        if self.is_blur_on:
            frame = detect_and_blur_faces(frame)

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image)
            self.canvas.image = image

        self.root.after(10, self.update)

    def __del__(self):
        self.cap.release()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceBlurApp(root)
    root.mainloop()
