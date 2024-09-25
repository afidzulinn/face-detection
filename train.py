from ultralytics import YOLO
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import multiprocessing

def main():
    model = YOLO('pretrained-model/yolov8n-face.pt')

    results = model.train(
        data='dataset.yaml',
        epochs=15,
        imgsz=640,
        batch=16,
        plots=True,
        device='cuda',
        name='yolov8_face_custom',
        workers=4
    )


if __name__ == '__main__':
    multiprocessing.freeze_support() 
    main()