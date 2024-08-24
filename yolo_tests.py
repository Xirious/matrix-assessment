from ultralytics import YOLO

model_best_str = "/home/student/git/personal/matrix/data/Human Forklift Objects Full.v1i.yolov5pytorch/runs/detect/train2/weights/best.pt"
data_str ="/home/student/git/personal/matrix/data/Human Forklift Objects Full.v1i.yolov5pytorch/data.yaml"

# Load a pretrained YOLOv8 model
model = YOLO(model_best_str)

metrics = model.val(data=data_str, split='test')

print(metrics.results_dict)