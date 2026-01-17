 ### Collect the data:

 1. data_collection.py

 python3 data_collection.py --data-root ./data --session-name half_manual --save-interval 0.5

 2. label_from_photos.py

 python3 label_from_photos.py --session ./data/half_manual --k 0.9 --deadzone 0.07 --steering-limit 0.6 --throttle 0.2
 
 
 
 ### train the model:

 train_control_cnn.py

python3 train_control_cnn.py --train-dirs ./data/full_manual_big ./data/full_manual_big1 ./data/straight ./data/curves_main ./data/wave_full ./data/half_manual --val-dirs ./data/full_manual ./data/full_manual_dark --out-dir ./models --aug-strength 0.8 --dropout 0.4


### autonomous driving:

drive_autonomous.py

python3 drive_autonomous.py


### evaluation model performance

eval_model.py

python3 eval_model.py


