 ### Collect the data:

 data_collection.py

 python3 data_collection.py --data-root ./data --session-name run1 --save-interval 0.5
 
 
 
 ### train the model:

 train_control_cnn.py

python3 train_control_cnn.py --train-dirs ./data/run_map_straight ./data/run_map_curves_main ./data/new_run_manual2 ./data/new_run_manual1 ./data/new_run_manual --val-dirs ./data/run_manual --out-dir ./models --aug-strength 0.8 --dropout 0.6


### autonomous driving:

drive_autonomous.py

python3 drive_autonomous.py


### evaluation model performance

eval_model.py

python3 eval_model.py