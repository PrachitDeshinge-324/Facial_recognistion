import cv2 as cv
import numpy as np
import time
from insightface.app import FaceAnalysis

# Corrected line
# Corrected line for your Mac
app = FaceAnalysis(name='buffalo_l', providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def open_video(video_path):
    """Open the video file and return the video capture object."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    return cap

def calculate_fps(frame_count, total_fps, start_time, current_time):
    """Calculate the current FPS and average FPS."""
    current_fps = 1 / (current_time - start_time) if (current_time - start_time) > 0 else 0
    total_fps += current_fps
    average_fps = total_fps / frame_count
    return current_fps, average_fps, total_fps

def display_fps(frame, current_fps, average_fps):
    """Overlay the FPS and Average FPS on the frame."""
    cv.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.putText(frame, f"Avg FPS: {average_fps:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def process_video(cap):
    """Process the video, frame by frame, and display FPS."""
    frame_count = 0
    total_fps = 0
    prev_time = 0 # Use a more descriptive name for the start_time of the loop

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # --- Start of processing for this frame ---
        start_time = time.time()
        
        frame_count += 1

        # Detect faces in the image
        faces = app.get(frame)

        # Draw bounding boxes on the faces
        rimg = app.draw_on(frame, faces)

        # --- End of processing for this frame ---
        current_time = time.time()
        
        # Calculate FPS based on the processing time for this frame
        processing_time = current_time - start_time
        current_fps = 1 / processing_time if processing_time > 0 else 0
        
        total_fps += current_fps
        average_fps = total_fps / frame_count

        # Overlay FPS on the result image (rimg)
        display_fps(rimg, current_fps, average_fps)

        # Show the frame with detections and FPS overlay
        cv.imshow('Video with FPS', rimg)

        if cv.waitKey(1) & 0xFF == ord('q'): # Use waitKey(1) for smoother video playback
            break


def main():
    video_path = '../Facial Recognision/video/03_09_2025_face_recognition.mp4'
    
    # Open the video
    cap = open_video(video_path)
    if cap is None:
        return

    # Process the video
    process_video(cap)

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
