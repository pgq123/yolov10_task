# coding:utf-8
from ultralytics import YOLOv10


# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/v10/yolov10b.yaml"

# 数据集配置文件
data_yaml_path = 'datasets/Data/data.yaml'

# 预训练模型
pre_model_name = 'weights/yolov10b.pt'

if __name__ == '__main__':
    # 加载预训练模型
    model = YOLOv10(model_yaml_path).load(pre_model_name)
    # 训练模型
    model.train(data=data_yaml_path,
                cache=False,
                imgsz=1024,
                epochs=150,
                batch=8,
                device='0',
                )