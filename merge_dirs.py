import os
import shutil

def merge_directories(source_dirs, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for source_dir in source_dirs:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                source_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)
                
                # Handling file name collisions
                if os.path.exists(dest_file_path):
                    base, extension = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(os.path.join(dest_dir, f"{base}_{counter}{extension}")):
                        counter += 1
                    dest_file_path = os.path.join(dest_dir, f"{base}_{counter}{extension}")
                
                shutil.copy2(source_file_path, dest_file_path)

# Example usage:
source_dirs = ["data1", "data2", "data3", "data4"]
dest_dir = "fictitious"
merge_directories(source_dirs, dest_dir)
