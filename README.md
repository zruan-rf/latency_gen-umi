# Camera latency characterization

## venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Data recording 
In one window run ```main.py``` to start camera capturing
In another window run ```ur_sine.py``` to move joint

## Post processing
Put the original data folder in ```latency_gen-umi/```

### Image batch process -- april tag
In ```batch_head_apriltag_video.py``` line 7-8, change the folder name to the target data folder
```python
INPUT_DIR = Path("/home/roboforce/Desktop/latency_gen-umi/<data_folder_name>/images/head")
REALSENSE_CSV = Path("/home/roboforce/Desktop/latency_gen-umi/<data_folder_name>/realsense_log.csv")
```
Run with 
```bash
python batch_head_apriltag_video.py 
```
As the output you will get a *head_apriltag_processed.mp4* and *head_apriltag_processed.csv*

### Latency calculation
Go to ```latency_shift.ipynb```

In **Cell 2**, modify the data folder name so that it points to the correct joint csv
```python
joint_data = pd.read_csv("<data_folder_name>/joint_log_left.csv")
```

If you wanted to remove plateaus in the beginning and at the end, you can crop the data by uncommenting this line in **Cell 2** and set the starting and ending time 
```python
# merged = merged[(merged["monotonic_ns"] > start_time) & (merged["monotonic_ns"] < end_time)].reset_index(drop=True)
```