import customtkinter as ctk
import tkinter as tk
from tkinter.filedialog import askopenfilename
import time
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2


fruit_calories = {
    'apple': 52,            # per 100g
    'banana': 89,           # per 100g
    'dragon fruit': 60,     # per 100g
    'guava': 68,            # per 100g
    'oren': 43,             # per 100g (assuming "oren" refers to orange)
    'pear': 57,             # per 100g
    'pineapple': 50,        # per 100g
    'sugar apple': 73       # per 100g
}

model = YOLO('fruits.pt')

root = ctk.CTk()
root.geometry('800x700')
root.title("fruits calories app")

root.grid_columnconfigure(0, weight=6)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(0, weight=6)
root.grid_rowconfigure(1, weight=1)

is_camera_open = False
video_paused = False


fruits_detected = []
sum_calories = []

def open_image():
    global file_path
    file_path = askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.png;*.gif;*.ppm;*.pgm;*jpg;*jpeg")])
    text_box.delete("1.0", ctk.END)
    text_box.insert(ctk.END, file_path)

    if file_path:
        load_and_display_image(file_path)

def load_and_display_image(file_path):
    global image
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (500, 500))
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image=image)
    canvas.create_image(0, 0, anchor="nw", image=image)


# put your video path here
cap2 = cv2.VideoCapture(r"P:\python_test\data\fruits.mp4")

def video_detect():
    global  cap2

    ret, frame = cap2.read()

    result = model.predict(frame, imgsz=864, conf=0.5)
    ploted_frame = result[0].plot()
    # print(ploted_frame.shape)

    img = cv2.cvtColor(ploted_frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (500, 500))
    img = ImageTk.PhotoImage(Image.fromarray(img))

    canvas.img = img
    canvas.create_image(0, 0, anchor="nw", image=img)
    root.after(5, video_detect)


def open_real_time():
    global cap, is_camera_open
    cap = cv2.VideoCapture(1)
    is_camera_open = True
    real_time()

def pause_video():
    global video_paused
    video_paused = not video_paused

def close_real_time():
    global is_camera_open, cap
    is_camera_open = False
    cap.release()
    root.quit()
    root.destroy()

def real_time():
    global cap, is_camera_open, video_paused

    if is_camera_open:
        if not video_paused:
            ret, frame = cap.read()

            result = model.predict(frame, imgsz=864, conf=0.5)
            ploted_frame = result[0].plot()
            # print(ploted_frame.shape)

            img = cv2.cvtColor(ploted_frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (500, 500))
            img = ImageTk.PhotoImage(Image.fromarray(img))

            canvas.img = img
            canvas.create_image(0, 0, anchor="nw", image=img)

    root.after(10, real_time)

def clear_frame():
    for widget in scorl_frame.winfo_children():
        widget.destroy()

def start():
    global fruits_detected
    clear_frame()

    frame = cv2.imread(file_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = model.predict(file_path, imgsz=864, conf=0.5)
    all_classes = result[0].names
    for indx, boxes in enumerate(result[0].boxes.xyxy):
        x1 = int(boxes[0])
        y1 = int(boxes[1])
        x2 = int(boxes[2])
        y2 = int(boxes[3])

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        frame = cv2.putText(frame,
                            str(all_classes[int(list(result[0].boxes.cls)[indx])]),
                            (x1+20, y1-10), 
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA)
        fruits_detected.append(f"{all_classes[int(list(result[0].boxes.cls)[indx])]} -> {fruit_calories[str(all_classes[int(list(result[0].boxes.cls)[indx])])]}")
        sum_calories.append(fruit_calories[str(all_classes[int(list(result[0].boxes.cls)[indx])])])


    fruits_label = ctk.CTkLabel(scorl_frame, text='\n'.join(fruits_detected), font=font)
    fruits_label.pack()

    sum_calories_label = ctk.CTkLabel(bottom_frame, text=f"sum calories\n{sum(sum_calories)}", font=('Modern', 25, 'bold'), corner_radius=30)
    sum_calories_label.grid(row=0, column=2)

    fruits_detected.clear()
    sum_calories.clear()
    scorl_frame.clipboard_clear()



    frame = cv2.resize(frame, (500, 500))
    frame = Image.fromarray(frame)
    frame = ImageTk.PhotoImage(frame)


    canvas.img = frame
    canvas.create_image(0, 0, anchor="nw", image=frame)

#-----------------------------------------------------------------------------------------------------------------


font = ('Minion Pro Med', 18)

left_frame = ctk.CTkFrame(root)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nswe')

right_frame = ctk.CTkFrame(root)
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nswe', rowspan=2)


bottom_frame = ctk.CTkFrame(root)
bottom_frame.grid(row=1, column=0, padx=10, pady=10, sticky='nswe')

text_box = ctk.CTkTextbox(right_frame, height=10)
text_box.pack(fill='x', pady=20, padx=5)
text_box.insert(index=ctk.END, text="Image path")

file_btn = ctk.CTkButton(right_frame, text='Uplode File', corner_radius=40, font=font, command=open_image)
file_btn.pack(padx=10)

stop_btn = ctk.CTkButton(right_frame, text='stop/close', hover_color='green', corner_radius=40,
                            font=font, command=close_real_time)
stop_btn.pack(pady=5,  side='bottom', )


pause_btn = ctk.CTkButton(right_frame, text='pause', hover_color='green', corner_radius=40,
                            font=font, command=pause_video)
pause_btn.pack(pady=5,  side='bottom')


realtime_btn = ctk.CTkButton(right_frame, text='Detect on real time', hover_color='green', corner_radius=40,
                            font=font, command=open_real_time)
realtime_btn.pack(pady=15, fill='x', side='bottom')


start_btn = ctk.CTkButton(right_frame, text='start', hover_color='green', corner_radius=40, font=font, command=start)
start_btn.pack(padx=30, pady=10)


video_btn = ctk.CTkButton(right_frame, text='Start video', hover_color='green', corner_radius=40, font=font, command=video_detect)
video_btn.pack(padx=30, pady=10, fill='x')


bp = ctk.CTkProgressBar(master=right_frame, height=10)
bp.pack(padx=30, side='bottom', pady=10)
bp.configure(mode='indeterminnate')
bp.start()


canvas = ctk.CTkCanvas(left_frame, bg='#2B2B2B')
canvas.pack(fill='both', expand=True)


scorl_frame = ctk.CTkScrollableFrame(bottom_frame)
scorl_frame.grid(row=0, column=0, sticky='nswe')


root.mainloop()



