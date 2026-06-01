from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import time
import os
import platform
import threading
import pyttsx3
from model_loader import load_model

app = Flask(__name__)

# ================= FOLDERS =================
os.makedirs("violations", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("Camera:", cap.isOpened())

# ================= MODEL =================
model = load_model()
print("Model loaded:", model.names)

# ================= GLOBAL =================
camera_running = False
last_capture_time = 0
capture_interval = 3
video_progress = 0

# ================= VOICE =================
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak_alert():
    try:
        engine.say("Helmet not detected")
        engine.runAndWait()
    except:
        pass

def beep():
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(1000, 500)
        else:
            print('\a')
    except:
        pass


# ================= WEBCAM =================
def generate_frames():
    global camera_running, last_capture_time

    while True:
        if not camera_running:
            time.sleep(0.1)
            continue

        success, frame = cap.read()
        if not success:
            continue

        results = model(frame)[0]
        no_helmet = False

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if "helmet" in label.lower() and "without" not in label.lower():
                color = (0,255,0)
                text = f"Helmet {conf:.2f}"
            else:
                color = (0,0,255)
                text = f"No Helmet {conf:.2f}"
                no_helmet = True

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,text,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

        # 🚨 ALERT + SAVE
        if no_helmet:
            now = time.time()
            if now - last_capture_time > capture_interval:
                filename = f"violations/violation_{int(now)}.jpg"
                cv2.imwrite(filename, frame)
                last_capture_time = now

                beep()
                speak_alert()

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# ================= IMAGE UPLOAD =================
@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    path = os.path.join("uploads", file.filename)
    file.save(path)

    results = model(path)[0]
    frame = results.plot()

    out_path = os.path.join("outputs", "img_" + file.filename)
    cv2.imwrite(out_path, frame)

    return jsonify({"path": "/outputs/" + os.path.basename(out_path)})


# ================= VIDEO PROCESS =================
def process_video(input_path, output_path):
    global video_progress

    cap_vid = cv2.VideoCapture(input_path)
    total = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap_vid.get(3)), int(cap_vid.get(4))))

    count = 0

    while True:
        ret, frame = cap_vid.read()
        if not ret:
            break

        results = model(frame)[0]
        frame = results.plot()
        out.write(frame)

        count += 1
        if total > 0:
            video_progress = int((count / total) * 100)

    cap_vid.release()
    out.release()
    video_progress = 100


@app.route('/upload_video', methods=['POST'])
def upload_video():
    file = request.files['file']
    path = os.path.join("uploads", file.filename)
    file.save(path)

    out_path = os.path.join("outputs", "vid_" + file.filename)

    threading.Thread(target=process_video,
                     args=(path, out_path)).start()

    return jsonify({"path": "/outputs/" + os.path.basename(out_path)})


# ================= PROGRESS =================
@app.route('/progress')
def progress():
    return jsonify({"progress": video_progress})


# ================= SERVE FILES =================
@app.route('/outputs/<path:filename>')
def send_output(filename):
    return send_from_directory('outputs', filename)


# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera')
def start_camera():
    global camera_running
    camera_running = True
    return "started"


@app.route('/stop_camera')
def stop_camera():
    global camera_running
    camera_running = False
    return "stopped"


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
