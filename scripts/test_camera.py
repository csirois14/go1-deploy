import cv2

cap = cv2.VideoCapture(
    "udpsrc port=9201 ! "
    "application/x-rtp,media=video,encoding-name=H264 ! "
    "rtph264depay ! h264parse ! avdec_h264 ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink",
    cv2.CAP_GSTREAMER,
)

if not cap.isOpened():
    print("Failed to open GStreamer pipeline")
else:
    print("Success! GStreamer works.")
