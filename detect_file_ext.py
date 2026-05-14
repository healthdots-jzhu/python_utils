import os
import mimetypes
from pathlib import Path

def detect_and_add_extension(folder_path):
    """
    Detect file types and add extensions to files without them.
    Files < 20MB default to .jpg, larger files default to .mp4
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_path):
            continue
        
        # Skip files that already have an extension
        if '.' in filename:
            continue
        
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        file_size = os.path.getsize(file_path)
        
        # Determine extension
        if mime_type:
            ext = mimetypes.guess_extension(mime_type) or get_default_extension(file_size)
        else:
            ext = get_default_extension(file_size)
        
        # Rename file
        new_name = filename + ext
        new_path = os.path.join(folder_path, new_name)
        os.rename(file_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

def get_default_extension(file_size):
    """Return extension based on file size"""
    size_mb = file_size / (1024 * 1024)
    return ".JPG" if size_mb < 20 else ".MOV"

if __name__ == "__main__":
    folder = input("Enter folder path: ")
    if os.path.isdir(folder):
        detect_and_add_extension(folder)
    else:
        print("Invalid folder path")