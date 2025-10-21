import os

from ultralytics import YOLO
from clearml import Task

MODEL_FNAME_OR_FPATH = 'yolo11l.pt'
MODEL_BASENAME = 'yolo11l'
DATASET = 'posco_under_windoor_winsldr_onlyinst_w320i80_251016.yaml'

#----------------------------------------------------------#

TAG_PLAN = 'under'  # under/unit/building ...
TAG_TASK = 'det'  # det/seg ...
TAG_TARGET = 'windoor'  # windoor/jubu/room/wall/core ...

TEST_NAME = 'TRAIN_251021_B'

#----------------------------------------------------------#


task = Task.init(
        project_name=os.path.join(TAG_PLAN, TAG_TASK, TAG_TARGET),
        task_name=TEST_NAME,
        task_type=Task.TaskTypes.training  # 또는 inference, data_processing 등
    )

# Load a model
model = YOLO(f"{MODEL_BASENAME}.yaml")  # build a new model from YAML
model = YOLO(MODEL_FNAME_OR_FPATH)  # load a pretrained model (recommended for training)
model = YOLO(f"{MODEL_BASENAME}.yaml").load(MODEL_FNAME_OR_FPATH)  # build from YAML and transfer weights

# Train the model
results = model.train(
    data=DATASET, 
    epochs=100, 
    imgsz=960, 
    batch=64,
    workers=8,
    project=f'{TAG_PLAN}',
    name=f'{TAG_TARGET}/{TEST_NAME}',
    )
