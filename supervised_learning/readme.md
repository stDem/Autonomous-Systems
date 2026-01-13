 ### train the model:
 
 train_control_cnn.py


python3 train_control_cnn.py \
  --data-dirs ./data/run_manual ./data/run_map_straight ./data/run_map_curves_main \
  --out-dir ./models \
  --aug-strength 0.8 \
  --dropout 0.6
