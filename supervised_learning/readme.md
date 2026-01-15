 ### Collect the data:

 data_collection.py

 python3 data_collection.py --data-root ./data --session-name run1 --save-interval 0.5
 
 
 
 ### train the model:

 train_control_cnn.py

python3 train_control_cnn.py --train-dirs ./data/straight ./data/curves_main ./data/full_manual_dark ./data/curve_small_first ./datacurve_small_second ./data/curve_small_last --val-dirs ./data/full_manual --out-dir ./models --aug-strength 0.8 --dropout 0.4


### autonomous driving:

drive_autonomous.py

python3 drive_autonomous.py


### evaluation model performance

eval_model.py

python3 eval_model.py