# EFDRL
http://bobbybby.top/2020-06-14-frl/

## 运行环境：
windows、linux均可
在linux环境，如果内存小于1GB，可能无法训练64*64的地图。

## 依赖：
python>=3.7.4、pytorch=1.4.0、torchvision=0.5.0、scipy  
建议使用conda安装上述依赖

## 运行：
在main.py中修改参数后，windows、linux启动方式均是直接运行main.py

## 输出：
linux下可以重定向输出，使用start_nohup.bash可以将输出重定向至根目录frl.log，实验最终结果输出至“all_mark”文件。（mark为自定义标志）  
每种地图每种模式输出一个最优结果至results目录  
每种地图每种模式每个网络的参数自动保存至weights目录

## 目录结构：
data：地图数据，每个文件包含8000张地图  
results：实验结果，每种情况一个文件，自动命名  
weights：保存的网络参数，每种情况每个网络一个文件，自动命名

## 文件说明：
conv：实现卷积网络的same-padding模式  
model：定义网络结构  
main：负责统筹训练任务，整理输出结果。  
agent：负责计划联邦学习的本地任务，类似智能体。 
dqn-network：负责训练网络，类似智能体的一个计算模块。  
replay-memory：负责经验缓存，是DQN需要的额外模块。  
environment：负责模拟真实环境，是强化学习必备的一个模块。  
![structure](https://github.com/BobbyBBY/EFDRL/blob/master/assert/structure.png)
