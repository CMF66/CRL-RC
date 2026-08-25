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

