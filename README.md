# PyTorch-YOLOv3-Tutorial
PyTorch-YOLOv3代码阅读-改进。
原项目地址：https://github.com/eriklindernoren/PyTorch-YOLOv3

更新：
1. 删除tensorflow的引用；
2. 修改测试未检测到目标时报错bug；
3. 修改dataset数据格式支持voc数据集；
4. 修改matplot循环显示内存泄漏。

环境安装：
pytorch>=1.3
torchvision>=0.5


conda install pytorch=1.5 torchvision=0.6 cudatoolkit=10.1
pip install opencv-python  matplotlib==2.2.4 terminaltables pillow tqdm