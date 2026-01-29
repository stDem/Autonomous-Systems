 ### Collect the data:

 1. data_collection.py

 python3 data_collection.py --data-root ./data --session-name new_full10 --save-interval 0.5

 2. label_from_photos.py

 python3 label_from_photos.py --session ./data/new_full10 --k 0.9 --deadzone 0.07 --steering-limit 0.9 --throttle 0.2
 
 ### train the model:

 train_control_cnn.py

python3 train_control_cnn.py --train-dirs ./data/full_manual ./data/full_manual_dark ./data/full_manual_big ./data/straight ./data/curves ./data/curves_main ./data/wave_full ./data/half_manual ./data/turn ./data/turn_more --val-dirs ./data/full_path --out-dir ./models --aug-strength 0.8 --dropout 0.2

python3 train_control_cnn.py --train-dirs ./data/new_full ./data/new_full2 ./data/new_full4 ./data/new_full5 ./data/new_full6 ./data/new_full7 ./data/new_full8 ./data/new_full10 --val-dirs ./data/new_full3 ./data/new_full9 --out-dir ./models --aug-strength 0.8 --dropout 0.2 --patience 20


### autonomous driving:

drive_autonomous.py

python3 drive_autonomous.py

### autonomous driving with object detection (YOLOv5):

drive_autonomous_yolo.py

python3 drive_autonomous_yolo.py

### evaluation model performance

eval_model.py

python3 eval_model.py

### testing motors' steering:

test.py

python3 test.py


### training YOLO5:

yolov5 -> train.py

cd yolov5

python train.py \
  --img 416 \
  --batch 16 \
  --epochs 150 \
  --data ../Autonomous-Systems/supervised_learning/datasets/od/data.yaml \
  --weights yolov5n.pt \
  --name od_yolov5n
