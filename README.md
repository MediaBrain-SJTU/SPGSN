# SPGSN
[ECCV2022] The source codes of 'Skeleton-parted graph scattering networks for 3D human motion prediction'. ECCV 2022

# Dependencies
Python 3.6

Pytorch 0.3.1.

progress 1.5

# Training commands
`python3 main_3d.py --data_dir "[Path To Your H36M data]/h3.6m/dataset/" --input_n 10 --output_n 10 --dct_n 15 --exp [where to save the log file]`

`python main_cmu_3d.py --data_dir_cmu "[Path To Your CMU data]/cmu_mocap/" --input_n 10 --output_n 25 --dct_n 30 --exp [where to save the log file]`

`python main_3dpw_3d.py --data_dir_3dpw "[Path To Your 3DPW data]/3DPW/sequenceFiles/" --input_n 10 --output_n 30 --dct_n 35 --exp [where to save the log file]`

# Citing
If you use our code, please cite our work

`
@inproceedings{li2022Skeleton,

  title={Skeleton-parted graph scattering networks for 3D human motion prediction},
  
  author={Li, Maosen and Chen, Siheng and Zhang, Zijing and Xie, Lingxi and Tian, Qi and Zhang, Ya},
  
  booktitle={ECCV},
  
  year={2022}
}
`
