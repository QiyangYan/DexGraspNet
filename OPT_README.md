## Quick Example
`quick_example.ipynb' in 'grasp_generation'

## Convert from isaac gym formate to dexgraspnet formate: `data/dexycb_inspire_convert.ipynb'
1. grasp_poses
2. output_path

## Optimize
1. n_iter
2. grasp_file
3. data_root_path
4. understand argument in `main_inspire_batch.py`
```
conda activate dexgraspnet
python grasp_generation/main_inspire_batch.py --fix_wrist --batch_size 1
```

## Pose and Optimized Pose Visualization
1. check argument in `visualize_result_optimize.py`
```
conda activate dexgraspnet
cd grasp_generation
python tests/visualize_result_optimize.py
```

## Convert to isaac gym compatible formate: `data/back2dexycb_inspire_convert.ipynb'
1. global_grasp_poses_dict
2. result_path 
3. output_path 