import cv2
import numpy as np
from smbus2 import SMBus
from time import sleep

# Initialize I2C for MLX90614
class MLX90614:
    def __init__(self, address=0x5A, bus=1):
        self.bus = SMBus(bus)
        self.address = address
    
    def read_temperature(self):
        data = self.bus.read_word_data(self.address, 0x07)
        return data * 0.02 - 273.15  # Convert raw data to Celsius

sensor = MLX90614()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect mask based on color heuristic
def detect_mask(frame, face):
    (x, y, w, h) = face
    face_roi = frame[y:y+h, x:x+w]

    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

    # Define a mask color range (e.g., blue or green masks)
    lower_mask_color = np.array([35, 50, 50])  # Adjust as needed
    upper_mask_color = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_mask_color, upper_mask_color)

    # Calculate the percentage of the face region covered by mask color
    mask_ratio = cv2.countNonZero(mask) / (w * h)

    return "Mask Detected" if mask_ratio > 0.2 else "Mask Not Detected"

# Initialize the Raspberry Pi camera
camera = cv2.VideoCapture(0)

try:
    print("Starting system. Press Ctrl+C to stop.")
    while True:
        # Capture a frame from the camera
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame")
            continue

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        mask_status = "No Face Detected"
        for face in faces:
            # Detect mask for each face
            mask_status = detect_mask(frame, face)

            # Draw a rectangle around the face
            (x, y, w, h) = face
            color = (0, 255, 0) if "Mask" in mask_status else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, mask_status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Read temperature
        temp = sensor.read_temperature()

        # Display temperature and mask status
        print(f"Temperature: {temp:.2f}°C, Status: {mask_status}")
        cv2.putText(frame, f"Temp: {temp:.2f}°C", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show the camera feed
        cv2.imshow("Mask Detection", frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Sleep for a second
        sleep(1)

except KeyboardInterrupt:
    print("System stopped by user.")

finally:
    # Release resources
    camera.release()
    cv2.destroyAllWindows()
