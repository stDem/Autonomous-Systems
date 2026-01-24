 ### Collect the data:

 1. data_collection.py

 python3 data_collection.py --data-root ./data --session-name sign_warning --save-interval 0.5

 2. label_from_photos.py

 python3 label_from_photos.py --session ./data/wave_full --k 0.9 --deadzone 0.07 --steering-limit 0.9 --throttle 0.2
 
 
 
 ### train the model:

 train_control_cnn.py

python3 train_control_cnn.py --train-dirs ./data/full_manual ./data/full_manual_dark ./data/full_manual_big ./data/straight ./data/curves ./data/curves_main ./data/wave_full ./data/half_manual ./data/turn --val-dirs ./data/full_path --out-dir ./models --aug-strength 0.8 --dropout 0.2


### autonomous driving:

drive_autonomous.py

python3 drive_autonomous.py


### evaluation model performance

eval_model.py

python3 eval_model.py


