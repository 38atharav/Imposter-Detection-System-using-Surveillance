import os
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_folder, fps=10):
    try:
        os.makedirs(output_folder, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video {video_path}")
            return False

        frame_count = 0
        success = True
        
        while success:
            success, frame = cap.read()
            if not success:
                break
                
            if frame_count % int(cap.get(cv2.CAP_PROP_FPS)/fps) == 0:
                frame = cv2.resize(frame, (224, 224))
                cv2.imwrite(f"{output_folder}/frame_{frame_count:04d}.jpg", frame)
            
            frame_count += 1
        
        cap.release()
        return frame_count > 0
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return False

# Process all videos with verification
for class_name in ['Fighting', 'Shooting','Explosion','RoadAccidents','Normal']:
    video_dir = os.path.abspath(f'Dataset/{class_name}')
    output_dir = os.path.abspath(f'frames/{class_name}')
    
    print(f"\nProcessing {class_name} videos...")
    for video_file in tqdm(os.listdir(video_dir)):
        video_path = os.path.join(video_dir, video_file)
        video_id = video_file.split('.')[0]
        video_output_dir = os.path.join(output_dir, video_id)
        
        success = extract_frames(video_path, video_output_dir)
        if not success:
            print(f"Failed to process: {video_file}")

print("\nFrame extraction complete. Verifying...")
for class_name in ['Fighting', 'Shooting','Explosion','RoadAccidents','Normal']:
    class_dir = os.path.abspath(f'frames/{class_name}')
    print(f"{class_name}: {len(os.listdir(class_dir))} videos extracted")