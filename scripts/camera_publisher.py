import cv2
import numpy as np
import subprocess
import threading
import queue
import socket
import time
import signal
import sys
import zmq
import json
from datetime import datetime


class ZMQCameraPublisher:
    def __init__(self, cam_id=1, width=640, height=480, ip_last_segment="164", zmq_port=5555, publish_rate=30):
        self.width = width
        self.height = height
        self.cam_id = cam_id
        self.ip_last_segment = ip_last_segment
        self.process = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.tcp_port = 5000 + cam_id
        self.zmq_port = zmq_port
        self.publish_rate = publish_rate  # FPS

        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{zmq_port}")
        print(f"ZeroMQ publisher bound to port {zmq_port}")

        # Statistics
        self.frame_count = 0
        self.published_count = 0

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
        frame_size = self.width * self.height * 3

        # Wait for GStreamer to start
        time.sleep(3)

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", self.tcp_port))
            sock.settimeout(1.0)

            print(f"Connected to GStreamer TCP port {self.tcp_port}")

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

                    self.frame_count += 1

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
                        continue
                except Exception as e:
                    if self.running:
                        print(f"Error reading frame: {e}")
                    break

            sock.close()

        except Exception as e:
            print(f"Error connecting to TCP: {e}")

    def zmq_publisher(self):
        """Publish frames via ZeroMQ"""
        frame_interval = 1.0 / self.publish_rate
        last_publish_time = 0

        while self.running:
            current_time = time.time()

            # Rate limiting
            if current_time - last_publish_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue

            try:
                frame = self.frame_queue.get_nowait()

                # Encode frame as JPEG for efficient transmission
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                # Create message with metadata
                message = {
                    "cam_id": self.cam_id,
                    "timestamp": current_time,
                    "frame_count": self.published_count,
                    "width": self.width,
                    "height": self.height,
                    "encoding": "jpeg",
                }

                # Send metadata first, then image data
                self.socket.send_string(json.dumps(message), zmq.SNDMORE)
                self.socket.send(buffer.tobytes())

                self.published_count += 1
                last_publish_time = current_time

                if self.published_count % 100 == 0:
                    print(f"Published {self.published_count} frames (received {self.frame_count})")

            except queue.Empty:
                time.sleep(0.001)
            except Exception as e:
                if self.running:
                    print(f"Error publishing frame: {e}")
                time.sleep(0.01)

    def start(self):
        """Start the camera and publishing"""
        cmd = self.get_gstreamer_command()
        print(f"Starting GStreamer: {' '.join(cmd)}")

        try:
            self.process = subprocess.Popen(cmd, stderr=subprocess.PIPE, bufsize=0)

            self.running = True

            # Start frame reading thread
            self.frame_thread = threading.Thread(target=self.frame_reader)
            self.frame_thread.daemon = True
            self.frame_thread.start()

            # Start ZMQ publishing thread
            self.publisher_thread = threading.Thread(target=self.zmq_publisher)
            self.publisher_thread.daemon = True
            self.publisher_thread.start()

            print(f"Camera started successfully!")
            print(f"Publishing on ZMQ port {self.zmq_port} at {self.publish_rate} FPS")
            return True

        except Exception as e:
            print(f"Error starting camera: {e}")
            return False

    def stop(self):
        """Stop the camera stream"""
        print("Stopping camera...")
        self.running = False

        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

        # Close ZMQ socket
        self.socket.close()
        self.context.term()

        print(f"Camera stopped. Published {self.published_count} frames total.")

    def run_headless(self):
        """Run without display (for server/robot use)"""
        if not self.start():
            print("Failed to start camera")
            return

        print("Camera running headless. Press Ctrl+C to stop.")

        # Setup signal handler for clean exit
        def signal_handler(sig, frame):
            print("\nStopping camera...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            while self.running:
                time.sleep(1)
                # Print status every 10 seconds
                if self.published_count > 0 and self.published_count % (self.publish_rate * 10) == 0:
                    print(f"Status: {self.published_count} frames published, {self.frame_count} frames received")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()

    def demo_with_display(self):
        """Demo with local display (for testing)"""
        if not self.start():
            print("Failed to start camera")
            return

        print("Press 'q' to quit")
        print("Frames are being published via ZeroMQ while displaying locally")

        # Setup signal handler for clean exit
        def signal_handler(sig, frame):
            print("\nStopping camera...")
            self.stop()
            cv2.destroyAllWindows()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            last_display_time = 0
            display_interval = 1.0 / 30  # 30 FPS display

            while True:
                current_time = time.time()

                if current_time - last_display_time >= display_interval:
                    try:
                        frame = self.frame_queue.get_nowait()
                        cv2.imshow(f"Camera {self.cam_id} (Publishing on ZMQ {self.zmq_port})", frame)
                        last_display_time = current_time
                    except queue.Empty:
                        pass

                # Press 'q' to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ZeroMQ Camera Publisher")
    parser.add_argument("--cam-id", type=int, default=1, help="Camera ID (1-5)")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZeroMQ port")
    parser.add_argument("--fps", type=int, default=30, help="Publishing frame rate")
    parser.add_argument("--headless", action="store_true", help="Run without display")
    parser.add_argument("--ip-segment", default="164", help="Last segment of IP address")

    args = parser.parse_args()

    print(f"=== ZeroMQ Camera Publisher ===")
    print(f"Camera ID: {args.cam_id}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"ZeroMQ Port: {args.zmq_port}")
    print(f"Publishing Rate: {args.fps} FPS")
    print(f"Headless Mode: {args.headless}")

    # Create camera publisher
    cam = ZMQCameraPublisher(
        cam_id=args.cam_id,
        width=args.width,
        height=args.height,
        zmq_port=args.zmq_port,
        publish_rate=args.fps,
        ip_last_segment=args.ip_segment,
    )

    if args.headless:
        cam.run_headless()
    else:
        cam.demo_with_display()
