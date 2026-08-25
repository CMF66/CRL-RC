# Clothing-Robust Representation Learning with Relation Calibration for Cloth-Changing Person Re-Identification

### Environment

- Python == 3.12
- PyTorch == 2.7.0
- faiss-gpu == 1.12.0      


### Training


# LTCC
python main.py --gpu_devices 0 --dataset ltcc --dataset_root DATASET_ROOT --dataset_filename LTCC-reID --save_dir SAVE_DIR --save_checkpoint

# Celeb-reID
python main.py --gpu_devices 0 --dataset celeb --dataset_root DATASET_ROOT --dataset_filename Celeb-reID --num_instances 4 --save_dir SAVE_DIR --save_checkpoint

# PRCC
python main.py --gpu_devices 0 --dataset prcc --dataset_root DATASET_ROOT --dataset_filename PRCC --save_dir SAVE_DIR --save_checkpoint

--dataset_root : replace DATASET_ROOT with your dataset root path

--save_dir: replace SAVE_DIR with the path to save log file and checkpoints


### Evaluation

python main.py --gpu_devices 0 --dataset DATASET --dataset_root DATASET_ROOT --dataset_filename DATASET_FILENAME --resume RESUME_PATH --save_dir SAVE_DIR --evaluate

--dataset: replace DATASET with the dataset name

--dataset_filename: replace DATASET_FILENAME with the folder name of the dataset

--resume: replace RESUME_PATH with the path of the saved checkpoint

### Acknowledgement

Some related work can be found from the following 
(https://github.com/QizaoWang/FIRe-CCReID)
