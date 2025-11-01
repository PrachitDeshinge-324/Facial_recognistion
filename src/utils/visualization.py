"""Visualization utilities for face recognition system."""

import cv2 as cv
import numpy as np


def draw_face_info(frame, face, name, confidence):
    """Draw face bounding box and recognition info on frame.
    
    Args:
        frame: Video frame
        face: Face object from InsightFace
        name: Recognized name
        confidence: Recognition confidence score
    """
    # Get bounding box coordinates
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    
    # Choose colors based on recognition status
    if name == "Unknown":
        box_color = (0, 0, 255)  # Red for unknown
        text_color = (0, 0, 255)
    else:
        box_color = (0, 255, 0)  # Green for recognized
        text_color = (0, 255, 0)
    
    # Draw bounding box
    cv.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    
    # Prepare text
    text = f"{name} ({confidence:.2f})"
    
    # Calculate text position
    text_size = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1] + 10
    
    # Draw text background
    cv.rectangle(frame, 
                (text_x, text_y - text_size[1] - 5), 
                (text_x + text_size[0] + 5, text_y + 5), 
                box_color, -1)
    
    # Draw text
    cv.putText(frame, text, (text_x + 2, text_y - 2), 
              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def display_fps(frame, current_fps, average_fps):
    """Display FPS information on frame.
    
    Args:
        frame: Video frame
        current_fps: Current FPS
        average_fps: Average FPS
    """
    cv.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv.putText(frame, f"Avg FPS: {average_fps:.2f}", (10, 60), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)


def display_system_info(frame, num_faces, threshold):
    """Display system information on frame.
    
    Args:
        frame: Video frame
        num_faces: Number of detected faces
        threshold: Recognition threshold
    """
    cv.putText(frame, f"Faces: {num_faces}", (10, 90), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv.putText(frame, f"Threshold: {threshold}", (10, 120), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)