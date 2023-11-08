import tkinter as tk
from ultralytics import YOLO
import cv2
import math
import tkinter as tk
from tkinter import filedialog

# Function to be called when Button 1 is clicked
def on_button1_click():
    label1.config(text="Image")
    # Tạo hộp thoại mở file để chọn ảnh
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.WEBP")])

    # Đọc ảnh từ đường dẫn file
    img = cv2.imread(file_path)

    model = YOLO(r"D:\DA_XLA\runs\detect\train3\weights\best.pt")

    classNames = ["pothole"]
    class_colors = {'pothole': (0, 0, 255)}  # Màu đỏ

    # Doing detections using YOLOv8
    results = model(img)

    # Once we have the results we will loop through them and we will have the bounding boxes for each of the result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls]
            label = f'{class_name} {conf}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            color = class_colors[class_name]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.imwrite(r"D:\DA_XLA\hinh.jpg", img)
    cv2.destroyAllWindows()

# Function to be called when Button 2 is clicked
def on_button2_click():
    label2.config(text="Video")
    # Tạo hộp thoại mở file để chọn video
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])

    cap = cv2.VideoCapture(file_path)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    output_path = r"D:\DA_XLA\video\detected_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 60, (frame_width, frame_height))

    model = YOLO(r"D:\DA_XLA\runs\detect\train3\weights\best.pt")

    classNames = ["pothole"]
    class_colors = {
        'pothole': (0, 0, 255)}  # màu đỏ
    while True:
        success, img = cap.read()
        if not success:
            break

        # Doing detections using YOLOv8 frame by frame
        # stream = True will use the generator and it is more efficient than normal
        results = model(img, stream=True)

        # Once we have the results we will loop through them and we will have the bounding boxes for each of the result
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                color = class_colors[class_name]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        out.write(img)
        cv2.imshow("VD", img)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()

# Function to be called when Button 3 is clicked
def on_button3_click():
    label3.config(text="Real time")
    from ultralytics import YOLO
    import cv2
    import math
    import torch

    cap = cv2.VideoCapture(0)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 200, (frame_width, frame_height))

    model = YOLO(r"D:\DA_XLA\runs\detect\train3\weights\best.pt")
    model.to(device)  # Chuyển mô hình lên GPU

    classNames = ["pothole"]
    class_colors = {
        'pothole': (0, 0, 255)}  # màu đỏ

    while True:
        success, img = cap.read()
        if not success:
            break

        # Doing detections using YOLOv8 frame by frame
        # stream = True will use the generator and it is more efficient than normal
        results = model(img, stream=True)

        # Once we have the results we will loop through them and we will have the bounding boxes for each of the result
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                color = class_colors[class_name]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        out.write(img)
        cv2.imshow("LIve", img)
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break
    out.release()
    cap.release()
    cv2.destroyAllWindows()

# Create a main window
window = tk.Tk()

# Set the window title
window.title("Pothole Detection")

# Create labels at the top
label1 = tk.Label(window, text="Image")
label2 = tk.Label(window, text="Video")
label3 = tk.Label(window, text="Real-time")

# Place the labels at the top
label1.grid(row=0, column=0)
label2.grid(row=0, column=1)
label3.grid(row=0, column=2)

# Create buttons in the center
button1 = tk.Button(window, text="Detect", command=on_button1_click)
button2 = tk.Button(window, text="Detect", command=on_button2_click)
button3 = tk.Button(window, text="Detect", command=on_button3_click)

# Place the buttons in the center
button1.grid(row=1, column=0)
button2.grid(row=1, column=1)
button3.grid(row=1, column=2)

# Start the Tkinter main loop
window.mainloop()
