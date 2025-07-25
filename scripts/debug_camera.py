import cv2
import subprocess
import sys


def check_gstreamer():
    """Check if GStreamer is installed and working"""
    try:
        result = subprocess.run(["gst-launch-1.0", "--version"], capture_output=True, text=True)
        print(f"GStreamer version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("GStreamer not found in system")
        return False


def check_opencv_gstreamer():
    """Check OpenCV GStreamer support"""
    build_info = cv2.getBuildInformation()

    print("OpenCV version:", cv2.__version__)

    # Look for GStreamer in build info
    gst_found = False
    for line in build_info.split("\n"):
        if "GStreamer" in line:
            print(line.strip())
            if "YES" in line:
                gst_found = True

    return gst_found


def test_simple_pipeline():
    """Test with a simple pipeline"""
    pipeline = "udpsrc port=9201 ! application/x-rtp,media=video,encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"

    print(f"Testing pipeline: {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ Failed to open with CAP_GSTREAMER")

        # Try without specifying backend
        cap = cv2.VideoCapture(pipeline)
        if not cap.isOpened():
            print("❌ Failed to open without backend specification")
            return False
        else:
            print("✅ Opened without backend specification")
    else:
        print("✅ Opened with CAP_GSTREAMER")

    # Try to read a frame
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"✅ Successfully read frame: {frame.shape}")
        cap.release()
        return True
    else:
        print("❌ Failed to read frame")
        cap.release()
        return False


def test_with_address():
    """Test with full address"""
    pipeline = "udpsrc address=192.168.123.164 port=9201 ! application/x-rtp,media=video,encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"

    print(f"Testing with address: {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ Failed to open")
        return False

    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"✅ Success: {frame.shape}")
        cap.release()
        return True
    else:
        print("❌ Failed to read frame")
        cap.release()
        return False


if __name__ == "__main__":
    print("=== Camera Debug Script ===\n")

    print("1. Checking GStreamer installation...")
    if not check_gstreamer():
        print("Please install GStreamer: sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-*")
        sys.exit(1)

    print("\n2. Checking OpenCV GStreamer support...")
    if not check_opencv_gstreamer():
        print("OpenCV doesn't have GStreamer support!")
        print("Try: pip install opencv-contrib-python")

    print("\n3. Testing simple pipeline...")
    if test_simple_pipeline():
        print("Simple pipeline works!")

    print("\n4. Testing with address...")
    if test_with_address():
        print("Full pipeline works!")

    print("\nDebug complete!")
