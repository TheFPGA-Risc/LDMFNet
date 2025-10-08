from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO(r"D:\CDP-YOLO\4090\ultralytics\ultralytics\cfg\models\model\yolov11n-two+DOIM.yaml")  # 初始化模型

    model.train(data=r"D:\CDP-YOLO\FLIR-align-3class\mydata.yaml", batch=16,
                epochs=300, project='runs/train4070/FLIR', name='yolov11n-two+DOIM-SDLoss',
                amp=False,
                workers=8,
                optimizer='SGD',  # Optimizer
                lr0=0.01,
                patience = 100,
                )  # 训练
    # model = YOLO(model=r"D:\CDP-YOLO\4090\ultralytics\runs\train4070\FLIR\yolov11n-two+DOIM-3stage-SDLoss\weights\last.pt")
    # model.train(resume=True)

