import os
import json
import argparse
import cv2
from ultralytics import YOLO

def detect_hands(model_path, input_dir, output_dir, logs_dir, confidence_threshold):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    class_mapping = {
        0: 'gloved_hand',
        1: 'bare_hand',
    }

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting detection...")

    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {filename}. Skipping.")
            continue

        results = model(img, conf=confidence_threshold)

        log_data = {
            "filename": filename,
            "detections": []
        }

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = round(float(box.conf[0]), 2)

                cls_id = int(box.cls[0])
                label = class_mapping.get(cls_id, f"unknown_class_{cls_id}")
                
                if label == 'person':
                    continue

                log_data["detections"].append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

                color = (0, 255, 0) if label == 'gloved_hand' else (0, 0, 255)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                text = f"{label}: {confidence}"
                
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        output_image_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_image_path, img)

        log_filename = os.path.splitext(filename)[0] + '.json'
        log_file_path = os.path.join(logs_dir, log_filename)
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        
        print(f"Processed {filename}. Annotated image and log saved.")

    print("\nDetection complete for all images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gloved vs. Ungloved Hand Detection Script")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pt model file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the folder containing input .jpg images.")
    parser.add_argument("--output_dir", type=str, default="output", help="Folder to save annotated images.")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Folder to save JSON detection logs.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections.")
    
    args = parser.parse_args()
    
    detect_hands(args.model_path, args.input_dir, args.output_dir, args.logs_dir, args.confidence)