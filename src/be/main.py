from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import os
from sqlalchemy import Column, Integer, String, create_engine, ForeignKey, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
from starlette.middleware.sessions import SessionMiddleware
from typing import Optional
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import mediapipe as mp
import asyncio
import base64
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
import grpc
from src.fe.streaming_pb2 import PushTextRequest, PopImageRequest
from src.fe.streaming_pb2_grpc import StreamingStub

camera = cv2.VideoCapture(0)
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
app = FastAPI()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "..", "fe", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "..", "fe", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

SQLALCHEMY_DATABASE_URL = "sqlite:///./users.db"  # Relative path, creates 'users.db' in current directory
# SQLALCHEMY_DATABASE2_URL = "sqlite:///./data_table.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app.add_middleware(SessionMiddleware, secret_key="YOUR_SECRET_KEY")

Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

capturing = False
captured_keypoints = []
current_word = None
current_user_id = None

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)

    # New column to denote admin or not
    is_admin = Column(Integer, default=0)  # 0 = not admin, 1 = admin


class DataTable(Base):
    __tablename__ = "data_table"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    word = Column(String, nullable=False)
    numpy_array = Column(LargeBinary, nullable=False)
    npy_file = Column(String, nullable=False)

# Create tables if not exist
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -----------------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """ Public home page. """
    return templates.TemplateResponse("text_to_image.html", {"request": request})

# -----------------------------
# REGISTER
# -----------------------------
@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db)
):
    # Check if username or email already exists
    existing_user = db.query(User).filter((User.username == username) | (User.email == email)).first()
    if existing_user:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error_msg": "Username or Email already exists!",
                "username": username,
                "email": email
            }
        )

    # Check password match
    if password != confirm_password:
        return templates.TemplateResponse(
            "register.html",
            {
                "request": request,
                "error_msg": "Passwords do not match!",
                "username": username,
                "email": email
            }
        )

    # Create new user
    hashed_pw = get_password_hash(password)
    new_user = User(username=username, email=email, password_hash=hashed_pw)
    db.add(new_user)
    db.commit()

    return templates.TemplateResponse(
        "login.html",
        {"request": request, "msg": "Registration successful! Please log in."}
    )

# -----------------------------
# LOGIN
# -----------------------------
@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login", response_class=HTMLResponse)
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    global current_user_id, current_is_admin

    user = db.query(User).filter(User.username == username).first()
    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error_msg": "Invalid username or password."}
        )

    if not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error_msg": "Invalid username or password."}
        )
    request.session["username"] = user.username
    # Set global user info
    current_user_id = user.id
    current_is_admin = (user.is_admin == 1)  # or user.is_admin for boolean

    return RedirectResponse(url="/welcome", status_code=302)


@app.get("/logout")
def logout(request: Request):
    request.session.pop("username", None)
    return RedirectResponse(url="/", status_code=302)

@app.get("/welcome", response_class=HTMLResponse)
def welcome(request: Request):
    # In a real app, you'd check session or JWT token to ensure user is authenticated
    if current_user_id is None:
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("home.html", {
        "request": request,
        "welcome_msg": "You are logged in!"
    })
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])
def gen_frames():
    """
    Generator function that:
      1. Captures frames from the webcam.
      2. Uses Mediapipe to detect hand landmarks.
      3. Draws the landmarks onto the frame.
      4. Encodes the frame as JPEG and yields it for streaming.
    """
    with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
        while True:
            success, frame = camera.read()
            if not success:
                # If there's an issue grabbing a frame, you could yield a blank or break
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            if capturing:
                keypoints = extract_keypoints(results)
                captured_keypoints.append(keypoints)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', image)
            if not ret:
                continue  # If encoding fails, skip this frame

            # Build a 'multipart/x-mixed-replace' response
            frame_bytes = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

@app.get("/data", response_class=HTMLResponse)
def data_page(request: Request):
    if current_user_id is None:
        # If user is not logged in, redirect to /login
        return RedirectResponse(url="/login", status_code=302)

    # Otherwise, render data.html
    return templates.TemplateResponse("data.html", {"request": request})
@app.get("/manage", response_class=HTMLResponse)
def manage_page(request: Request, db: Session = Depends(get_db)):
    # Check if user is logged in
    if current_user_id is None:
        return RedirectResponse(url="/login", status_code=302)

    # Query data_table for the current user
    user_data = db.query(DataTable).filter(DataTable.user_id == current_user_id).all()

    # Render the manage.html template, passing the user's data
    return templates.TemplateResponse(
        "manage.html",
        {
            "request": request,
            "data_rows": user_data
        }
    )

@app.get("/manage/edit/{record_id}", response_class=HTMLResponse)
def edit_page(record_id: int, request: Request, db: Session = Depends(get_db)):
    if current_user_id is None:
        return RedirectResponse(url="/login", status_code=302)

    data_record = db.query(DataTable).filter_by(id=record_id).first()
    if not data_record:
        return {"error": "Record not found"}

    # Check if user owns it or is admin
    if data_record.user_id != current_user_id and not current_is_admin:
        return {"error": "Not authorized to edit this record"}

    # Render a template with a form
    return templates.TemplateResponse(
        "edit_record.html",
        {
            "request": request,
            "record_id": record_id,
            "existing_word": data_record.word
        }
    )


@app.post("/manage/edit/{record_id}", response_class=HTMLResponse)
def edit_record_post(record_id: int, request: Request, db: Session = Depends(get_db), word: str = Form(...)):
    if current_user_id is None:
        return RedirectResponse(url="/login", status_code=302)

    data_record = db.query(DataTable).filter_by(id=record_id).first()
    if not data_record:
        return {"error": "Record not found"}

    # Check if user owns it or is admin
    if data_record.user_id != current_user_id and not current_is_admin:
        return {"error": "Not authorized to edit this record"}

    # Update the word field
    data_record.word = word
    db.commit()

    return RedirectResponse(url="/manage", status_code=302)

@app.post("/manage/delete/{record_id}")
def delete_record(record_id: int, db: Session = Depends(get_db)):
    if current_user_id is None:
        return {"error": "Not logged in"}

    data_record = db.query(DataTable).filter_by(id=record_id).first()
    if not data_record:
        return {"error": "Record not found"}

    # Check if user owns it or is admin
    if data_record.user_id != current_user_id and not current_is_admin:
        return {"error": "Not authorized to delete this record"}

    db.delete(data_record)
    db.commit()

    return RedirectResponse(url="/manage", status_code=302)

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
@app.post("/stop_camera")
def stop_camera():
    global camera
    if camera.isOpened():
        camera.release()
    return {"status": "Camera stopped"}

@app.post("/capture/start")
def capture_start(    request: Request,
    word: str = Form(...),
    db: Session = Depends(get_db)
):
    """
    Start capturing frames for a given word.
    We assume the user is logged in (somehow).
    """

    global capturing, captured_keypoints, current_word, current_user_id

    if current_user_id is None:
        current_user_id = 1
    capturing = True
    current_word = word
    captured_keypoints = []
    return templates.TemplateResponse(
        "data.html",
        {
            "request": request,
            "message": f"Capturing started for word: {current_word}",
        }
    )

@app.post("/capture/stop")
def capture_stop(db: Session = Depends(get_db)):
    """
    Stop capturing and save the .npy file.
    Then insert record into the data_table.
    """
    global capturing, captured_keypoints, current_word, current_user_id

    capturing = False

    if not current_word or not captured_keypoints:
        return {"status": "no data recorded or no word provided"}

    # Convert list to numpy array
    keypoints_array = np.array(captured_keypoints)

    # Build filename
    filename = f"{current_word + str(current_user_id)}.npy"
    # Save under a subfolder, e.g. "data_npy" if you want
    # For simplicity, just save in current directory
    array_bytes = keypoints_array.tobytes()
    np.save(filename, keypoints_array)

    # Insert record into DB
    db_record = DataTable(
        user_id=current_user_id,
        word=current_word,
        numpy_array=array_bytes,
        npy_file=filename
    )
    db.add(db_record)
    db.commit()

    # Reset or keep them for next time
    saved_word = current_word
    current_word = None
    captured_keypoints = []

    return {"status": "capturing stopped", "saved_file": filename, "word": current_word}

def get_grpc_stub():
    channel = grpc.insecure_channel('localhost:50051')
    return StreamingStub(channel)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stub = get_grpc_stub()

    pop_image_response = stub.PopImage(PopImageRequest(time_stamp=""))

    try:
        while True:
            # Receive text from the client
            text = await websocket.receive_text()
            # Push text to gRPC service
            push_image_response = stub.PushText(PushTextRequest(text=text, time_stamp=""))
            # Start receiving images
            for response in pop_image_response:
                if response.image:
                    # Convert bytes to base64
                    base64_image = base64.b64encode(response.image).decode('utf-8')
                    print("Show image in fe")
                    await websocket.send_text(f"data:image/jpeg;base64,{base64_image}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)