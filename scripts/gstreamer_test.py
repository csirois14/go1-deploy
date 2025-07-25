import cv2
import numpy as np
import subprocess
import threading
import queue
import socket
import time
import signal
import sys


class TCPGStreamerCamera:
    def __init__(self, cam_id=1, width=640, height=480, ip_last_segment="164"):
        self.width = width
        self.height = height
        self.cam_id = cam_id
        self.ip_last_segment = ip_last_segment
        self.process = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.tcp_port = 5000 + cam_id  # Use different ports for different cameras

    def get_gstreamer_command(self):
        """Generate GStreamer command using TCP"""
        udp_ports = [9201, 9202, 9203, 9204, 9205]
        port = udp_ports[self.cam_id - 1]

        cmd = [
            "gst-launch-1.0",
            f"udpsrc",
            f"address=192.168.123.{self.ip_last_segment}",
            f"port={port}",
            "!",
            "application/x-rtp,media=video,encoding-name=H264",
            "!",
            "rtph264depay",
            "!",
            "h264parse",
            "!",
            "avdec_h264",
            "!",
            "videoconvert",
            "!",
            "videoscale",
            "!",
            f"video/x-raw,width={self.width},height={self.height},format=BGR",
            "!",
            "tcpserversink",
            f"host=127.0.0.1",
            f"port={self.tcp_port}",
        ]

        return cmd

    def frame_reader(self):
        """Read frames from TCP socket"""
        frame_size = self.width * self.height * 3  # BGR = 3 bytes per pixel

        # Wait for GStreamer to start
        time.sleep(3)

        try:
            # Connect to GStreamer TCP server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", self.tcp_port))
            sock.settimeout(1.0)  # 1 second timeout

            print(f"Connected to TCP port {self.tcp_port}")

            while self.running:
                try:
                    # Read exact frame size
                    raw_frame = b""
                    while len(raw_frame) < frame_size and self.running:
                        chunk = sock.recv(frame_size - len(raw_frame))
                        if not chunk:
                            print("Connection closed by GStreamer")
                            return
                        raw_frame += chunk

                    if len(raw_frame) != frame_size:
                        continue

                    # Convert to numpy array
                    frame = np.frombuffer(raw_frame, dtype=np.uint8)
                    frame = frame.reshape((self.height, self.width, 3))

                    # Apply camera-specific transformations
                    if self.cam_id == 1:
                        frame = cv2.flip(frame, -1)

                    # Add to queue (remove old frame if queue is full)
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        try:
                            self.frame_queue.get_nowait()  # Remove old frame
                            self.frame_queue.put_nowait(frame)  # Add new frame
                        except queue.Empty:
                            pass

                except socket.timeout:
                    if self.running:
                        continue  # Timeout is normal, just retry
                except Exception as e:
                    if self.running:
                        print(f"Error reading frame: {e}")
                    break

            sock.close()

        except Exception as e:
            print(f"Error connecting to TCP: {e}")

    def start(self):
        """Start the camera stream"""
        cmd = self.get_gstreamer_command()
        print(f"Starting GStreamer: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(cmd, stderr=subprocess.PIPE, bufsize=0)

            self.running = True

            # Start frame reading thread
            self.frame_thread = threading.Thread(target=self.frame_reader)
            self.frame_thread.daemon = True
            self.frame_thread.start()

            print(f"Camera started successfully on TCP port {self.tcp_port}!")
            return True

        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def get_frame(self):
        """Get the latest frame"""
        try:
            frame = self.frame_queue.get_nowait()
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self):
        """Stop the camera stream"""
        self.running = False

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

        print("Camera stopped")

    def demo(self):
        """Demo function to display video"""
        if not self.start():
            print("Failed to start camera")
            return

        print("Press 'q' to quit")
        print("Waiting for frames... (this may take a few seconds)")

        # Setup signal handler for clean exit
        def signal_handler(sig, frame):
            print("\nStopping camera...")
            self.stop()
            cv2.destroyAllWindows()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            frame_count = 0
            no_frame_count = 0

            while True:
                ret, frame = self.get_frame()

                if ret:
                    cv2.imshow(f"Camera {self.cam_id}", frame)
                    frame_count += 1
                    no_frame_count = 0

                    if frame_count % 30 == 0:  # Print every 30 frames
                        print(f"Frames received: {frame_count}")
                else:
                    no_frame_count += 1
                    if no_frame_count > 100:  # If no frames for a while
                        print("No frames received for a while...")
                        no_frame_count = 0

                # Press 'q' to quit
                key = cv2.waitKey(30) & 0xFF  # Increased wait time
                if key == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    print("=== TCP GStreamer Camera ===")

    # Test if gst-launch-1.0 is available
    try:
        result = subprocess.run(["gst-launch-1.0", "--version"], capture_output=True, text=True)
        print("GStreamer available ✅")
    except FileNotFoundError:
        print("GStreamer not found ❌")
        sys.exit(1)

    # Create and run camera
    cam = TCPGStreamerCamera(cam_id=1, width=640, height=480)
    cam.demo()
