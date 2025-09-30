"""Real-time face recognition from video or camera."""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import cv2 as cv
import tqdm

from src.core.detection import FaceDetector
from src.core.recognition import FaceRecognizer
from src.core.database import FaceDatabase
from src.utils.video import open_video, create_video_writer, compute_fps_metrics
from src.utils.visualization import draw_face_info, display_fps, display_system_info
from src.config.paths import Video_Path, Output_Video_Path


def process_video(video_path, output_path, display=True):
    """Process video with face recognition."""
    
    # Initialize components
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    database_handler = FaceDatabase()
    
    # Load face database
    if database_handler.load():
        recognizer.set_database(database_handler.get_database())
    else:
        print("Running in detection-only mode (no recognition)")
    
    # Open video source
    try:
        cap = open_video(video_path)
    except ValueError as e:
        print(f"Error: {e}")
        return False
    
    # Set up output writer
    try:
        out = create_video_writer(cap, output_path)
    except ValueError as e:
        print(f"Error: {e}")
        cap.release()
        return False
    
    # Initialize metrics
    frame_count = 0
    total_fps = 0
    
    print("\nStarting face recognition...")
    if display:
        print("Press 'q' to quit")
    
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    with tqdm.tqdm(total=total_frames, desc="Processing Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Start timing
            start_time = time.perf_counter()
            frame_count += 1
            
            # Detect faces
            faces = detector.detect_faces(frame)
            
            # Process each face
            for face in faces:
                name, confidence = recognizer.recognize_face(face.embedding)
                draw_face_info(frame, face, name, confidence)
            
            # Calculate FPS
            current_time = time.perf_counter()
            current_fps, average_fps, total_fps = compute_fps_metrics(
                frame_count, total_fps, start_time, current_time
            )
            
            # Display info
            display_fps(frame, current_fps, average_fps)
            display_system_info(frame, len(faces), recognizer.threshold)
            
            # Write frame
            out.write(frame)
            
            # Display frame only if requested
            if display:
                cv.imshow('Face Recognition', frame)
                # Check for exit
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    if display:
        cv.destroyAllWindows()
    
    print(f"\nOutput video saved to: {output_path}")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run face recognition on video")
    parser.add_argument("--video", type=str, help="Path to video file (default: configured path)")
    parser.add_argument("--camera", type=int, help="Camera device index to use")
    parser.add_argument("--output", type=str, help="Output video path")
    parser.add_argument("--no-display", action="store_true", help="Disable frame display (headless mode)")
    args = parser.parse_args()
    
    # Determine video source
    if args.camera is not None:
        video_source = args.camera
        print(f"Using camera {args.camera}")
    elif args.video:
        video_source = args.video
        print(f"Using video file: {args.video}")
    else:
        video_source = Video_Path
        print(f"Using default video: {Video_Path}")
    
    # Determine output path
    output_path = args.output if args.output else Output_Video_Path
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Display mode
    display = not args.no_display
    if not display:
        print("Running in headless mode (no frame display)")
    
    # Process video
    success = process_video(video_source, output_path, display=display)
    
    if success:
        print("\n✅ Video processing completed successfully!")
    else:
        print("\n❌ Video processing failed")


if __name__ == "__main__":
    main()