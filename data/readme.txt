To run the codes, please first put the data into the 'data' folder.

Then, for Human3.6M, run:
   python main_3d.py --input_n 10 --output_n 10 --dct_n 15 --data_dir ./data/h3.6m/dataset/ --num_stage 10 --J 2

For CMU Mocap, run:
   python main_cmu_3d.py --input_n 10 --output_n 10 --dct_n 15 --data_dir ./data/h3.6m/dataset/ --num_stage 10 --J 2

For 3DPW, run:
   python main_3dpw_3d.py --input_n 10 --output_n 10 --dct_n 15 --data_dir ./data/h3.6m/dataset/ --num_stage 8 --J 2