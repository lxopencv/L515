# L515
使用L515测量带有aruco信息物品的实时距离
python 3.8
以上代码均为chatgpt-4生成
要实现这个功能，需要分为以下几个步骤：
生成Aruco标记并保存为图像文件
使用Intel RealSense L515相机捕获图像
检测Aruco标记并计算相机内外参数
将内外参数保存到YAML文件
首先，确保安装了以下所需的库：
pip install opencv-python
pip install opencv-contrib-python 3.4.8
pip install pyrealsense2
pip install yaml
