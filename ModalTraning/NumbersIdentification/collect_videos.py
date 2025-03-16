import os
import cv2

DATA_DIR = './video_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 11  # Adjust as needed
videos_per_class = 100 # Number of videos per class
video_duration = 2  # Duration of each video in seconds
fps = 1  # Frames per second
frame_width = int(cv2.VideoCapture(0).get(3))
frame_height = int(cv2.VideoCapture(0).get(4))

cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting videos for class {j}')

    for i in range(videos_per_class):
        print(f'Get ready for video {i + 1}/{videos_per_class}. Press "Q" to start.')

        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Press "Q" to Start Recording', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_filename = os.path.join(class_dir, f'video_{i}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

        print(f'Recording video {i + 1} for class {j}...')
        frame_count = 0
        max_frames = video_duration * fps

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            cv2.imshow('frame', frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to stop early
                break

        out.release()
        print(f'Video {i + 1} for class {j} saved.')

cap.release()
cv2.destroyAllWindows()
print('Data collection complete.')