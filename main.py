import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceDetector:
    def __init__(self):
        self.model = YOLO("model/best.onnx")
        logging.info("model YOLO dimuat")

    def detect(self, frame, conf_threshold=0.6):
        results = self.model(frame, conf=conf_threshold)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.item()
                cls = int(box.cls.item())
                detections.append((x1, y1, x2, y2, conf, cls))
        return detections

class FaceComparator:
    def __init__(self, reference_path, tolerance=0.6):
        self.reference_encoding = self.load_reference(reference_path)
        self.tolerance = tolerance

    @staticmethod
    def load_reference(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Foto referensi tidak ditemukan: {path}")
        
        reference_image = cv2.imread(path)
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(reference_image)
        
        if not encodings:
            raise ValueError(f"Tidak ada wajah yang ditemukan dalam foto referensi: {path}")
        
        logging.info(f"Wajah referensi berhasil dimuat dari: {path}")
        return encodings[0]

    def compare(self, face_image):
        if face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        if face_image.shape[0] < 20 or face_image.shape[1] < 20:
            return False

        try:
            face_encoding = face_recognition.face_encodings(face_image)
            if face_encoding:
                return face_recognition.compare_faces([self.reference_encoding], face_encoding[0], tolerance=self.tolerance)[0]
        except Exception as e:
            logging.warning(f"Error saat encoding wajah: {e}")
        
        return False


def process_video(video_path, face_detector, face_comparator, output_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Tidak dapat membuka video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    
    detections = []

    for frame_count in tqdm(range(total_frames), desc="Memproses video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = timedelta(seconds=frame_count/fps)
        frame_detections = face_detector.detect(frame)
        
        for (x1, y1, x2, y2, conf, cls) in frame_detections:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            if cls == 0:  # Face
                face_image = frame[y1:y2, x1:x2]
                
                if face_comparator.compare(face_image):
                    detections.append((current_time, "face"))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    time_str = (datetime.min + current_time).time().strftime('%H:%M:%S')
                    cv2.putText(frame, f"Match at {time_str}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            elif cls == 1:  # Mask
                detections.append((current_time, "mask"))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Mask", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # progress = int((frame_count / total_frames) * 100)
        # cv2.putText(frame, f"Progress: {progress}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.rectangle(frame, (10, 60), (10 + progress * 2, 80), (0, 255, 0), -1)
            
        out.write(frame)
    
    cap.release()
    out.release()
    return detections

def main():
    video_path = "video/Robbery at Spare parts Shop in Khairpur, Sindh, Pakistan _ CCTV _ Footage.mp4"
    reference_path = "images/1.jpg"
    output_path = "result/video-result.mp4"
    timestamp_path = "result/timestamps-result.txt"

    try:
        face_detector = FaceDetector()
        face_comparator = FaceComparator(reference_path)
        
        logging.info("Memulai pemrosesan video")
        detections = process_video(video_path, face_detector, face_comparator, output_path)
        
        logging.info("Pemrosesan video selesai")
        print("\nDeteksi wajah dan masker pada waktu:")
        
        # menyimpan timestamp 
        with open(timestamp_path, 'w') as f:
            f.write("Deteksi wajah dan masker pada waktu:\n")
            for time, detection_type in detections:
                time_str = (datetime.min + time).time().strftime("%H:%M:%S")
                output_str = f"{time_str} - {detection_type}"
                print(output_str)
                f.write(f"{output_str}\n")
        
        logging.info(f"Timestamp dan eteksi disimpan di {timestamp_path}")

    except FileNotFoundError as e:
        logging.error(f"File tidak ditemukan: {e}")
    except ValueError as e:
        logging.error(f"Error dalam pemrosesan: {e}")
    except Exception as e:
        logging.error({e})

if __name__ == "__main__":
    main()