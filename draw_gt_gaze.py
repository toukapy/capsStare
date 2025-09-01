import cv2
import csv
import os
import numpy as np

INPUT_VIDEO = "gaze_last_result.avi"
GT_CSV      = "ground_truth_arrows.csv"
OUTPUT_GT_VIDEO = "gaze_with_gt.avi"

# ============ Utilities ============
def load_annotations(csv_path):
    ann = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if not row: continue
                fi, ox, oy, ex, ey = map(int, row[:5])
                ann[fi] = (ox, oy, ex, ey)
    return ann

def save_annotations(csv_path, ann_dict):
    rows = []
    for fi, (ox, oy, ex, ey) in sorted(ann_dict.items()):
        rows.append([fi, ox, oy, ex, ey])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "origin_x", "origin_y", "end_x", "end_y"])
        w.writerows(rows)
    print(f"[SAVE] {len(rows)} annotations -> {csv_path}")

# ============ Frame-by-frame Annotator ============
def annotate_video_stepwise(
    video_path=INPUT_VIDEO,
    out_csv=GT_CSV,
    start_frame=0,
    step=1,            # annotate every 'step' frames (e.g., 1 = every frame, 5 = every 5th)
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ann = load_annotations(out_csv)

    # Find first frame to annotate (resume from last unannotated)
    current = start_frame
    if ann:
        # resume on the first frame >= start_frame that is not annotated AND matches the step
        existing = set(ann.keys())
        for i in range(start_frame, total, step):
            if i not in existing:
                current = i
                break

    points = []  # [(x,y), (x,y)]
    window = "Annotator (stepwise)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def on_click(event, x, y, flags, param):
        nonlocal points
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
                points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            points = []  # clear current

    cv2.setMouseCallback(window, on_click)

    help1 = "Left-click: ORIGIN then END | rmb: clear"
    help2 = "[n] next  [p] prev  [u] undo  [s] save  [j] jump  [q] quit"
    help3 = f"Annotating every {step} frame(s)."

    while True:
        if current < 0: current = 0
        if current >= total:
            print("Reached end of video.")
            break

        # Position on specific frame and grab exactly one frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current)
        ok, frame = cap.read()
        if not ok:
            print(f"Cannot read frame {current}.")
            break

        # Draw existing GT for this frame
        disp = frame.copy()
        if current in ann:
            ox, oy, ex, ey = ann[current]
            cv2.arrowedLine(disp, (ox, oy), (ex, ey), (0, 255, 0), 2, tipLength=0.2)
            cv2.circle(disp, (ox, oy), 3, (0, 255, 255), -1)

        # Draw temporary clicks
        if len(points) >= 1:
            cv2.circle(disp, points[0], 4, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.arrowedLine(disp, points[0], points[1], (0, 0, 255), 2, tipLength=0.2)

        # HUD
        cv2.putText(disp, f"Frame {current}/{total-1}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(disp, help1, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(disp, help2, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        cv2.putText(disp, help3, (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        cv2.imshow(window, disp)
        key = cv2.waitKey(0) & 0xFF  # BLOCK until a key is pressed

        if key == ord('q'):
            # optional: auto-save on quit
            save_annotations(out_csv, ann)
            break

        elif key == ord('s'):
            save_annotations(out_csv, ann)

        elif key == ord('u'):
            # undo current annotation
            if current in ann:
                del ann[current]
                print(f"[UNDO] Removed annotation for frame {current}")
            points = []

        elif key == ord('n'):
            # commit points if present
            if len(points) == 2:
                ann[current] = (points[0][0], points[0][1], points[1][0], points[1][1])
                print(f"[ADD] Frame {current}: {ann[current]}")
                points = []
            # advance by 'step'
            current += step

        elif key == ord('p'):
            # go to previous annotated step
            points = []
            current -= step

        elif key == ord('j'):
            # jump to frame number typed in console
            try:
                inp = input(f"Jump to frame [0..{total-1}]: ").strip()
                j = int(inp)
                if 0 <= j < total:
                    # snap to the closest step boundary if you want consistent cadence
                    if step > 1:
                        j = start_frame + ((j - start_frame) // step) * step
                    current = j
                    points = []
                else:
                    print("Invalid frame number.")
            except Exception as e:
                print("Jump cancelled.", e)

        else:
            # ignore other keys
            pass

    cap.release()
    cv2.destroyAllWindows()
    # final save
    save_annotations(out_csv, ann)

# ============ Renderer: burn GT arrows into a new video ============
def render_gt_video(input_video=INPUT_VIDEO, csv_path=GT_CSV, output_video=OUTPUT_GT_VIDEO):
    ann = load_annotations(csv_path)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx in ann:
            ox, oy, ex, ey = ann[idx]
            cv2.arrowedLine(frame, (ox, oy), (ex, ey), (0, 255, 0), 2, tipLength=0.2)
            cv2.circle(frame, (ox, oy), 3, (0, 255, 255), -1)
        out.write(frame)
        idx += 1

    cap.release()
    out.release()
    print(f"[RENDER] Saved GT-burned video -> {output_video}")

if __name__ == "__main__":
    # 1) Annotate stepwise (blocks until you finish)
    annotate_video_stepwise(
        video_path=INPUT_VIDEO,
        out_csv=GT_CSV,
        start_frame=0,
        step=1,          # change to 5/10 to annotate every Nth frame
    )

    # 2) Burn GT into a new video (optional)
    render_gt_video(INPUT_VIDEO, GT_CSV, OUTPUT_GT_VIDEO)
