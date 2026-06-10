import os
import cv2
import numpy as np
import csv
import argparse
from ultralytics import YOLO

# Command-line arguments
parser = argparse.ArgumentParser(description="Run YOLO + depth clustering on synced RGB-D PNGs.")
parser.add_argument("--rgbd_dir", required=True, help="Folder containing color_*.png and depth_*.png files")
parser.add_argument("--yolo_model", required=True, help="Path to YOLO model weights")
parser.add_argument("--output_csv", required=True, help="Output detection CSV path")
parser.add_argument("--conf", type=float, default=0.6, help="YOLO confidence threshold")
parser.add_argument("--n_clusters", type=int, default=3, help="Number of depth clusters for K-means")
parser.add_argument("--start_frame", type=int, default=0, help="Starting frame index for quick testing")
parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process for quick testing")
args = parser.parse_args()

# Folder containing synced RGB-D PNG data
raw_data_folder = args.rgbd_dir
output_csv_file = args.output_csv

# Load YOLO model
model = YOLO(args.yolo_model)

# PARAMETER FOR CLUSTERING
n_clusters = args.n_clusters

# Get all depth and color filenames
depth_files = sorted([f for f in os.listdir(raw_data_folder) if f.startswith("depth_") and f.endswith(".png")])
color_files = sorted([f for f in os.listdir(raw_data_folder) if f.startswith("color_") and f.endswith(".png")])

# Optional frame subset for quick testing
if args.max_frames is not None:
    depth_files = depth_files[args.start_frame:args.start_frame + args.max_frames]
    color_files = color_files[args.start_frame:args.start_frame + args.max_frames]
elif args.start_frame > 0:
    depth_files = depth_files[args.start_frame:]
    color_files = color_files[args.start_frame:]

# Ensure corresponding depth and color files exist
assert len(depth_files) == len(color_files), "Mismatch between depth and color files"

# Open CSV file and write headers
with open(output_csv_file, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Time (in ms)", "YOLO Class", "YOLO Confidence",
        "Center X", "Center Y",
        "Bounding Box X1", "Bounding Box Y1", "Bounding Box X2", "Bounding Box Y2",
        "Min Cluster Depth (mm)" # column for the depth from the closest cluster
    ])

    # Process frames one by one
    for depth_file, color_file in zip(depth_files, color_files):
        # Extract timestamp parts and convert to milliseconds
        filename_parts = depth_file.split("_")
        seconds_str = filename_parts[1]
        nanoseconds_str = filename_parts[2].split(".")[0]

        try:
            seconds = int(seconds_str)
            nanoseconds = int(nanoseconds_str)
            timestamp_ms = seconds * 1000 + nanoseconds / 1000000.0
        except ValueError:
            print(f"Warning: Could not parse timestamp from {depth_file}. Skipping frame.")
            continue

        timestamp = f"{timestamp_ms:.3f}" # Format to 3 decimal places for milliseconds

        # Load depth data
        depth_path = os.path.join(raw_data_folder, depth_file)
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # Load as 16-bit unchanged
        # depth_data = cv2.rotate(depth_data, cv2.ROTATE_180) # Remove comment if image is rotated (current images are not)

        # Load color image
        color_path = os.path.join(raw_data_folder, color_file)
        color_image = cv2.imread(color_path) # Load as 8-bit BGR
        # color_image = cv2.rotate(color_image, cv2.ROTATE_180) # Remove comment if image is rotated (current images are not)

        # Handle potential loading errors (e.g., if a file is corrupted)
        if color_image is None or depth_data is None:
            print(f"Warning: Could not load {color_file} or {depth_file}. Skipping frame.")
            continue

        # If depth_data is 3-channel (e.g., grayscale PNG loaded as RGB), convert to single channel
        if depth_data.ndim == 3:
            depth_data = depth_data[..., 0]

        # Get image dimensions for boundary checks
        h, w = depth_data.shape

        print(f"Processing frame at time: {timestamp} ms")

        # YOLOv8 Inference
        results = model(color_image, conf=args.conf) # Confidence threshold changeable

        # Parse YOLO results
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                # Ensure bounding box coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 # Center coordinates

                # DEPTH CALCULATION USING K-MEANS CLUSTERING 
                min_cluster_depth = -1 # Default value if clustering fails

                # Extract ROI for depth clustering
                # Ensure ROI is not empty
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    roi = depth_data[y1:y2, x1:x2]

                    # Get non-zero depth points within the ROI
                    ys_roi, xs_roi = np.nonzero(roi)
                    pts = roi[ys_roi, xs_roi].astype(np.float32).reshape(-1, 1)

                    # Perform K-means clustering only if enough points are available
                    if pts.size >= n_clusters:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                        _, labels, centers = cv2.kmeans(pts, n_clusters, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
                        centers = centers.flatten()

                        # Sort cluster centers by ascending depth
                        sorted_centers = np.sort(centers)

                        # The minimum cluster depth is the first element in the sorted centers
                        min_cluster_depth = sorted_centers[0]
                    else:
                        # Fallback if not enough points for clustering
                        # We might choose to use average depth here, or keep -1
                        if pts.size > 0:
                            min_cluster_depth = np.mean(pts)
                        else:
                            min_cluster_depth = -1
                yolo_class_name = model.names[int(class_id)] # YOLO class name

                # Write detection data to CSV file
                csv_writer.writerow([
                    timestamp, yolo_class_name, f"{confidence:.2f}",
                    cx, cy, x1, y1, x2, y2,
                    f"{min_cluster_depth:.2f}" # Write the new depth metric
                ])

                # Annotate the color image with bounding box and label
                # cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Updated label to show the new depth metric
                # label_text = f"{yolo_class_name} D:{min_cluster_depth:.0f}mm"
                # cv2.putText(color_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                #                 0.5, (0, 255, 0), 2)

        # Display annotated frame
        # cv2.imshow("YOLOv11 Detection", color_image)
        key = cv2.waitKey(1)
        if key == 27: # ESC key to break
            break

cv2.destroyAllWindows()
print("Processing complete. CSV saved successfully!")