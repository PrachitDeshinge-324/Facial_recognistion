# Imports
from pathlib import Path

# Video paths
Video_Path = Path("../Facial Recognision/video/03_09_2025_face_recognition.mp4")
Output_Dir = Path("output")
Output_Video_Path = Output_Dir / "03_09_2025_output_face_recognition.mp4"

# Database paths
Database_Dir = Path("database")
Database_Data_Path = Path("face_database")
Database_Path = Database_Dir / "face_database_antelopev2.pkl"
Metadata_Path = Database_Dir / "metadata_antelopev2.json"

# Logging paths
Log_Dir = Path("logs")
Log_File_Path = Log_Dir / "recognition_log.txt"

# Create directories if they don't exist
if not Output_Dir.exists():
    Output_Dir.mkdir(parents=True)

if not Database_Dir.exists():
    Database_Dir.mkdir(parents=True)
