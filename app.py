from flask import Flask, render_template, request, redirect, url_for, Response, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import time
from collections import deque
# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this!
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///alerts.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define LRCN Model Architecture
class LRCN(nn.Module):
    def __init__(self, num_classes=5, lstm_hidden=128, lstm_layers=1):
        super(LRCN, self).__init__()
        # CNN Feature Extractor
        base = mobilenet_v2(pretrained=False)
        self.cnn = nn.Sequential(*list(base.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # LSTM
        self.lstm = nn.LSTM(1280, lstm_hidden, lstm_layers, batch_first=True)
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden, num_classes)
        )
    
    def forward(self, x):
        batch, seq_len = x.shape[:2]
        c_in = x.view(batch * seq_len, *x.shape[2:])
        c_out = self.cnn(c_in)
        c_out = self.adaptive_pool(c_out)
        c_out = c_out.view(batch, seq_len, -1)
        lstm_out, _ = self.lstm(c_out)
        return self.fc(lstm_out[:, -1, :])

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LRCN().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Activity labels
class_to_label = {0: "Fighting", 1: "Shooting", 2: "Explosion",3:"RoadAccidents",4:"Normal"}
suspicious_classes = [0, 1,2,3]

# Email config (replace with your credentials)
EMAIL_CONFIG = {
    'GMAIL_USER': 'gharalsandesh1@gmail.com',
    'GMAIL_PASSWORD': 'zvvn bwue howo zymi',
    'ADMIN_EMAIL': 's28272821@gmail.com'
}

# User database (simplified - use proper DB in production)
users = {
    "admin": {"password": bcrypt.generate_password_hash("admin123").decode('utf-8')}
}

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(username):
    return User(username) if username in users else None

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    activity = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def send_email_alert(activity):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG['GMAIL_USER']
    msg['To'] = EMAIL_CONFIG['ADMIN_EMAIL']
    msg['Subject'] = f"ALERT: {activity} Detected"
    msg.attach(MIMEText(f"Activity detected: {activity} at {datetime.now()}", 'plain'))
    
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['GMAIL_USER'], EMAIL_CONFIG['GMAIL_PASSWORD'])
            server.send_message(msg)
    except Exception as e:
        print(f"Email failed: {e}")

    db.session.add(Alert(activity=activity))
    db.session.commit()

SEQUENCE_LENGTH = 16
FRAME_SKIP = 2                   
MIN_CONFIDENCE = 0.90            # Increased confidence threshold
ALERT_COOLDOWN = 10              # Longer cooldown between alerts
CONSENSUS_WINDOW = 5             # Number of consecutive detections required
VIDEO_FPS = 15                   

# Activity smoothing with deque
activity_buffer = deque(maxlen=CONSENSUS_WINDOW)

def process_frame(frame):
    """Enhanced frame preprocessing"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256, 256))  # Slightly larger before center crop
    frame = transform(frame)
    return frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frame_count = 0
    result = "No suspicious activity detected."
    last_alert_time = 0
    activity_buffer.clear()
    
    with app.app_context():
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue
            
            try:
                processed = process_frame(frame)
                sequence.append(processed)
                
                if len(sequence) > SEQUENCE_LENGTH:
                    sequence = sequence[-SEQUENCE_LENGTH:]
                
                if len(sequence) == SEQUENCE_LENGTH:
                    inputs = torch.stack(sequence).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(inputs)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        conf, pred = torch.max(probabilities, 1)
                        confidence = conf.item()
                        pred_class = pred.item()
                    
                    # Apply consensus voting
                    if confidence > MIN_CONFIDENCE:
                        activity_buffer.append(pred_class)
                    
                    current_time = time.time()
                    if (len(activity_buffer) == CONSENSUS_WINDOW and
                        current_time - last_alert_time > ALERT_COOLDOWN):
                        
                        # Get most common activity in buffer
                        consensus = max(set(activity_buffer), key=activity_buffer.count)
                        if consensus in suspicious_classes:
                            activity = class_to_label[consensus]
                            result = f"ALERT: {activity} detected (Confidence: {confidence:.2f}, Consensus: {CONSENSUS_WINDOW}/{CONSENSUS_WINDOW})"
                            send_email_alert(activity)
                            last_alert_time = current_time
                            activity_buffer.clear()
                            break
                    else:
                        return result
            
            except Exception as e:
                print(f"Video processing error: {str(e)}")
                continue
    
    cap.release()
    return result

def generate_frames():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
    sequence = []
    frame_count = 0
    last_alert_time = 0
    activity_buffer.clear()
    last_prediction = 2  # Start with "Normal"
    
    with app.app_context():
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                # Add visual feedback even for skipped frames
                text = f"Status: {class_to_label[last_prediction]}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue
            
            try:
                processed = process_frame(frame)
                sequence.append(processed)
                
                if len(sequence) > SEQUENCE_LENGTH:
                    sequence = sequence[-SEQUENCE_LENGTH:]
                
                detection_text = "Analyzing..."
                if len(sequence) == SEQUENCE_LENGTH:
                    inputs = torch.stack(sequence).unsqueeze(0).to(device)
                    with torch.no_grad():
                        outputs = model(inputs)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        conf, pred = torch.max(probabilities, 1)
                        confidence = conf.item()
                        pred_class = pred.item()
                        last_prediction = pred_class
                    
                    # Apply consensus voting
                    if confidence > MIN_CONFIDENCE:
                        activity_buffer.append(pred_class)
                    
                    current_time = time.time()
                    if (len(activity_buffer) == CONSENSUS_WINDOW and
                        current_time - last_alert_time > ALERT_COOLDOWN):
                        
                        consensus = max(set(activity_buffer), key=activity_buffer.count)
                        if consensus in suspicious_classes:
                            activity = class_to_label[consensus]
                            print(f"Consensus Alert: {activity} (Confidence: {confidence:.2f})")
                            send_email_alert(activity)
                            last_alert_time = current_time
                            activity_buffer.clear()
                
                    detection_text = f"{class_to_label[pred_class]} ({confidence:.2f})"
                
                # Enhanced visual feedback
                cv2.putText(frame, f"Status: {detection_text}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if (len(activity_buffer) >= CONSENSUS_WINDOW and 
                    activity_buffer.count(activity_buffer[0]) == CONSENSUS_WINDOW and
                    activity_buffer[0] in suspicious_classes):
                    
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                    cv2.putText(frame, f"ALERT: {class_to_label[activity_buffer[0]]}", 
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                continue

# Routes
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and bcrypt.check_password_hash(users[username]['password'], password):
            login_user(User(username))
            return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = process_video(filepath)
            return render_template('result.html', result=result)
    
    return render_template('upload.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime')
@login_required
def realtime():
    return render_template('realtime.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/alerts')
@login_required
def view_alerts():
    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    return render_template('alerts.html', alerts=alerts)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)