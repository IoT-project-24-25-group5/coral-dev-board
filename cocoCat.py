import asyncio
import base64
import json
import cv2
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
import numpy as np
import websockets
import time
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common, detect


web_socket = None  # global websocket connection
cst = None  # global camera stream track


class CameraStreamTrack(VideoStreamTrack):
    def __init__(
        self,
        device_index=1,
        model_path="tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite",
        labels_path="coco_labels.txt",
    ):
        super().__init__()
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {device_index}")

        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()

        self.labels = self.load_labels(labels_path)

        # For ping control
        self.last_ping_time = 0
        self.cat_detected_last_frame = False
        self.first_detected_at = 0
        self.ping_interval = 5  # seconds

    def load_labels(self, path):
        labels = {}
        with open(path, "r") as f:
            for i, line in enumerate(f.readlines()):
                label = line.strip()
                if label:
                    labels[i] = label
        return labels

    async def recv(self):
        global web_socket

        pts, time_base = await self.next_timestamp()

        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        input_size = common.input_size(self.interpreter)
        resized = cv2.resize(frame, input_size)

        common.set_input(self.interpreter, resized)
        self.interpreter.invoke()

        objs = detect.get_objects(self.interpreter, score_threshold=0.5)
        cats = [obj for obj in objs if obj.id == 16]  # COCO cat class id

        current_time = time.time()
        cat_detected_now = len(cats) > 0

        if cat_detected_now:
            if not self.cat_detected_last_frame:
                # New detection, record timestamp
                self.first_detected_at = current_time

            cat = max(
                cats,
                key=lambda o: (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]),
            )
            xmin, ymin, xmax, ymax = map(int, cat.bbox)
            label = self.labels.get(cat.id, "cat")
            score = cat.score

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label}: {score:.2f}",
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if (not self.cat_detected_last_frame) or (
                current_time - self.last_ping_time > self.ping_interval
            ):
                if web_socket is not None:
                    try:
                        await web_socket.send(
                            json.dumps(
                                {
                                    "type": "cat_detected",
                                    "first_detected_at": self.first_detected_at,
                                }
                            )
                        )
                        print(f"Ping sent: cat_detected at {self.first_detected_at}")
                        self.last_ping_time = current_time
                    except Exception as e:
                        print("Failed to send ping:", e)

            self.cat_detected_last_frame = True

        else:
            cv2.putText(
                frame,
                "No cat detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            self.cat_detected_last_frame = False
            self.last_ping_time = 0
            self.first_detected_at = 0

        video_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


async def cleanupRTC(pc):
    await pc.close()


async def startRTCpc(client_offer):
    pc = RTCPeerConnection()

    await pc.setRemoteDescription(RTCSessionDescription(**client_offer))
    global cst
    cst = CameraStreamTrack()
    pc.addTrack(cst)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    global web_socket
    await web_socket.send(
        json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    )

    @pc.on("connectionstatechange")
    def on_connection_state_change():
        print("Connection state:", pc.connectionState)
        if pc.connectionState in ("failed", "disconnected", "closed"):
            asyncio.create_task(cleanupRTC(pc))
            print("Connection closed")


async def handleWsMessage(message):
    loaded = json.loads(message)
    print("Received websocket message:", loaded)
    if loaded.get("type") == "offer":
        print("Received offer from browser")
        asyncio.create_task(startRTCpc(loaded))
    elif loaded["type"] == "getFrame":
        print("Received getFrame from browser")
        global cst
        if cst is None:
            cst = CameraStreamTrack()
        if cst is not None:
            ret, frame = cst.cap.read()
            frame = cv2.resize(frame, (640, 480))

            # Encode frame to JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            # Base64 encode JPEG bytes
            b64_frame = base64.b64encode(jpeg.tobytes()).decode('utf-8')

            # Create JSON message
            message = {
                "type": "frame",
                "frame": b64_frame
            }

            # Send as JSON string
            await web_socket.send(json.dumps(message))


async def connectWs():
    uri = "wss://iot.philippevoet.dev"
    async with websockets.connect(uri) as websocket:
        await websocket.send("devBoard")
        global web_socket
        web_socket = websocket
        while True:
            try:
                message = await websocket.recv()
                await handleWsMessage(message)
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                web_socket = None
                break


async def run():
    await connectWs()


if __name__ == "__main__":
    asyncio.run(run())
