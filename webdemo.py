import streamlit as st
import cv2
import numpy as np
import av
import time

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import tensorflow as tf

st.set_page_config(page_title="Driver Drowsiness Detection")

st.title("Driver Drowsiness Detection System")

# Load model once
model = tf.keras.models.load_model("model.keras")

# Load cascades
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


class VideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.start_time = None

    def recv(self, frame):

        frame = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        eye_rois = []

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            face_gray = gray[y:y+h, x:x+w]
            face_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(
                face_gray,
                scaleFactor=1.1,
                minNeighbors=4
            )

            eyes = eyes[:2]

            for (ex, ey, ew, eh) in eyes:

                eye = face_color[ey:ey+eh, ex:ex+ew]

                eye_rois.append(eye)

                cv2.rectangle(
                    face_color,
                    (ex, ey),
                    (ex + ew, ey + eh),
                    (0, 255, 0),
                    2
                )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

        font = cv2.FONT_HERSHEY_SIMPLEX

        if len(eye_rois) > 0:

            preds = []

            for eye in eye_rois:

                try:

                    img = cv2.resize(
                        eye,
                        (224, 224)
                    )

                    img = img / 255.0

                    img = np.expand_dims(
                        img,
                        axis=0
                    )

                    pred = model.predict(
                        img,
                        verbose=0
                    )[0][0]

                    preds.append(pred)

                except:
                    pass

            if len(preds) > 0:

                final_pred = sum(preds) / len(preds)

                if final_pred > 0.5:

                    status = "OPEN EYES"

                    self.start_time = None

                    cv2.putText(
                        frame,
                        status,
                        (100, 100),
                        font,
                        1.5,
                        (0, 255, 0),
                        2
                    )

                else:

                    status = "CLOSED EYES"

                    cv2.putText(
                        frame,
                        status,
                        (100, 100),
                        font,
                        1.5,
                        (0, 0, 255),
                        2
                    )

                    if self.start_time is None:

                        self.start_time = time.time()

                    elif time.time() - self.start_time > 2:

                        cv2.putText(
                            frame,
                            "SLEEP ALERT!",
                            (100, 200),
                            font,
                            1.5,
                            (0, 0, 255),
                            3
                        )

                        # Browser alarm placeholder
                        cv2.putText(
                            frame,
                            "ALARM ON",
                            (100, 260),
                            font,
                            1,
                            (0, 0, 255),
                            2
                        )

        else:

            cv2.putText(
                frame,
                "NO EYES DETECTED",
                (100, 100),
                font,
                1.2,
                (255, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(
            frame,
            format="bgr24"
        )


webrtc_streamer(
    key="drowsiness",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)