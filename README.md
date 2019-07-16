# 超分辨率任务
# 项目列表
  - ESRGAN_demo: ESRGAN的丐版实现，用于改进的一个基础模型 



PI (perceptual index)感知指数

来自三篇论文


$$
\text{Perceptual index} = \tfrac{1}{2} ((10 - \text{Ma}) + \text{NIQE}).
$$
**Ma** 来自论文http://xxx.itp.ac.cn/pdf/1612.05890v1

用于评估图像的感知质量

**NIQE** 来自论文https://ieeexplore.ieee.org/document/6353522/

https://blog.csdn.net/mazhitong1020/article/details/80415758

同样的评估图像感知质量的指标

**RMSE**评估图像失真度的指标



两个指标结合起来做图像质量感知的评估，在PI值相同时失真越小图像越好

具体参考https://blog.csdn.net/CHNguoshiwushuang/article/details/80556419

![1563198208423](https://www.pirm2018.org/img/regions.svg)

纵轴质量分 横轴失真度 成反比 

分数评估代码和[ PIRM Challenge](https://www.pirm2018.org/PIRM-SR.html)竞赛一样（matlab）

https://github.com/KwanWaiPang/NIQECalculation