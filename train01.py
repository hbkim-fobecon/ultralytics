from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(
    data="posco_under_windoor_winsldr_onlyinst_w320i80_251016.yaml", 
    epochs=1, 
    imgsz=320, 
    batch=128,
    workers=8,
    project='under',
    name='det/windoor/test-0',
    )
