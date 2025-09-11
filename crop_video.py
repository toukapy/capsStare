import cv2

input_video = "gaze_output.avi"
output_video = "gaze_output_cropped.avi"

start_frame = 2500   # skip the first 100 frames
end_frame = None    # or put a number like 1000 to stop earlier

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")

out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx >= start_frame and (end_frame is None or frame_idx <= end_frame):
        out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"Video saved to {output_video}")
