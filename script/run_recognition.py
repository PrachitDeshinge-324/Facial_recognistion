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


def process_video(video_path, output_path, display=True, batch_size=4, preprocess=False, skip_frames=1):
    """Process video with face recognition using frame batching."""
    
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
    is_camera = isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit())
    try:
        # Use threading for cameras for better performance
        cap = open_video(video_path, use_threading=is_camera)
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
    process_start_time = time.perf_counter()
    
    # Variables for frame skipping
    last_faces = []
    last_results = []
    
    print("\nStarting face recognition...")
    if is_camera:
        print("Real-time mode enabled (Threaded Capture)")
        # For real-time, we force batch_size to 1 to minimize latency
        batch_size = 1
        
    if display:
        print("Press 'q' to quit")
    
    # Get total frames if available (not for camera)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) if not is_camera else 0
    
    pbar = tqdm.tqdm(total=total_frames, desc="Processing Video") if total_frames > 0 else None
    
    should_exit = False
    
    while cap.isOpened() and not should_exit:
        # Read batch of frames
        batch_frames = []
        
        # Start timing for batch
        batch_start_time = time.perf_counter()
        
        # Read a batch of frames
        for _ in range(batch_size):
            if not cap.isOpened():
                break
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.perf_counter()
            frame_count += 1
            batch_frames.append((frame, frame_start_time))
        
        if not batch_frames:
            break  # No frames read, end of video
            
        # Process batch of frames
        all_frames = [item[0] for item in batch_frames]
        batch_faces = []
        batch_results = []
        
        # If skipping frames, we only detect on the first frame of the batch (or every Nth frame)
        # But since we are batching, we might as well detect on all if batch_size is small.
        # For real-time (batch_size=1), we use skip_frames logic.
        
        if is_camera and skip_frames > 1:
            # Real-time skipping logic
            frame, frame_start_time = batch_frames[0] # batch_size is 1
            
            if frame_count % skip_frames == 0:
                # Run detection
                faces = detector.detect_faces(frame, preprocess=preprocess)
                if faces:
                    results = recognizer.recognize_faces(faces)
                else:
                    results = []
                
                last_faces = faces
                last_results = results
            else:
                # Use last known results
                faces = last_faces
                results = last_results
                
            batch_faces.append(faces)
            batch_results.append(results)
            
        else:
            # Standard batch processing (process all frames)
            batch_faces = [detector.detect_faces(frame, preprocess=preprocess) for frame in all_frames]
            
            for faces in batch_faces:
                if faces:
                    recognition_results = recognizer.recognize_faces(faces)
                    batch_results.append(recognition_results)
                else:
                    batch_results.append([])
        
        batch_end_time = time.perf_counter()
        batch_process_time = batch_end_time - batch_start_time
        
        # Process results and display frames one by one
        for i, ((frame, frame_start_time), faces, results) in enumerate(zip(batch_frames, batch_faces, batch_results)):
            # Apply results to faces
            for face, (name, confidence) in zip(faces, results):
                draw_face_info(frame, face, name, confidence)
            
            # Calculate FPS
            frame_process_time = batch_process_time / len(batch_frames)
            current_time = frame_start_time + frame_process_time
            current_fps, average_fps = compute_fps_metrics(
                frame_count - (len(batch_frames) - i - 1), 
                process_start_time, 
                frame_start_time, 
                current_time
            )
            
            # Display info
            display_fps(frame, current_fps, average_fps)
            display_system_info(frame, len(faces), recognizer.threshold)
            
            if preprocess:
                cv.putText(frame, "Preprocess: ON", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Write frame
            out.write(frame)
            
            # Display frame only if requested
            if display:
                cv.imshow('Face Recognition', frame)
                # Check for exit
                if cv.waitKey(1) & 0xFF == ord('q'):
                    should_exit = True
                    break
            
            if pbar:
                pbar.update(1)
    
    # Cleanup
    if pbar:
        pbar.close()
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
    parser.add_argument("--batch-size", type=int, default=4, help="Number of frames to process in each batch")
    parser.add_argument("--preprocess", action="store_true", help="Enable preprocessing for outdoor/difficult conditions")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame (for real-time performance)")
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
    success = process_video(
        video_source, 
        output_path, 
        display=display, 
        batch_size=args.batch_size,
        preprocess=args.preprocess,
        skip_frames=args.skip_frames
    )
    
    if success:
        print("\n✅ Video processing completed successfully!")
    else:
        print("\n❌ Video processing failed")


if __name__ == "__main__":
    main()