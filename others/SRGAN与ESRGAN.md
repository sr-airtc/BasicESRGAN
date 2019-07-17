## SRGAN与ESRGAN

### SRGAN

#### 所用的损失函数

1. 内容损失（content loss） 

2. 对抗学习的损失函数（adversarial loss）

   

   ![](https://i.loli.net/2019/07/17/5d2e8d665dcf122298.png)

内容损失有两种

- 像素级的均方误差（之前的做法）
- 特征级的均方误差【这里称为感知损失（perceptual loss）】
  1. VGG提取图像的低级特征
  2. VGG提取图像的高级特征

本文得出的结论是用感知损失比较好

### ESRGAN（增强型的SRGAN）

#### 改进点

- 网络结构的改进（残差块改成残差和DenseNet结合的密集连接残差块）并且对残差用了一个0-1的缩放

- 去掉残差块中的bn层（会导致残影）

- 传统GAN中的D改为相对判别器RaD（不是绝对的判别图片的真假，而是相对真实度）参考RaGAN

- 对于SRGAN中的感知损失使用的激活以后的特征做mse改成激活前的特征（激活后变稀疏，监督效果差，并且会造成亮度差异）

- 网络插值（看不懂:joy:）

  

  ![](https://i.loli.net/2019/07/17/5d2e9266785a396173.png)

ESRGAN 

https://zhuanlan.zhihu.com/p/54473407

https://blog.csdn.net/qq_36556893/article/details/86418149

两者区别

https://blog.csdn.net/qxqsunshine/article/details/85223123

RaGAN

https://zhuanlan.zhihu.com/p/40403886