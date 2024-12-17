import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    for yaml_name in ['yolo11-FPSC-RGCSPELAN','yolo11-FPSC']:
        model = YOLO(f'/final/FRNet/models/innovations/{yaml_name}.yaml')
        model.train(data='/final/datasets/STBD-08/STBD.yaml',
                    cache=False,
                    imgsz=640,
                    epochs=300,
                    batch=32,
                    close_mosaic=0,
                    workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                    # device='0',
                    optimizer='SGD', # using SGD
                    # patience=0, # set 0 to close earlystop.
                    # resume=True, # 断点续训,YOLO初始化时选择last.pt
                    # amp=False, # close amp
                    # fraction=0.2,
                    project='runs/train/STBD-08',
                    name=yaml_name,
                    )