
# 检测图片的颜色丰富度

## 算法用途：
检测图片是否为颜色丰富，从而筛选单调的图片，目前的召回率比较高，可能对一些比较淡雅的艺术图片有误杀率，下一步将提高准确率
具体来说可以检测出 图表、聊天截图、表情包 等

## 算法思路：
计算颜色分布直方图，如果分布非常集中，那么说明颜色比较单调

## 算法步骤：
1. 统计 RGB（0～255） 三个通道的值分布，得到 0～255 每个数值出现的频率
2. 根据频率对0～255的数值进行排序，得出频率最高的5个值，相加为 top_5_sum
3. 计算top_5_sum 占总像素点的比例
4. 颜色丰富度 color_rich_degree = 1 - top_5_sum/(img_width \* img_height \* 3)

## Todo
对一定量的样本计算召回率、准确度等
根据样本结果，拟合出 颜色丰富度的公式，比如使用 1 - log(...) 等

根据这个算法来影响焦点图的计算

https://github.com/xiaohe10/imageLab


```python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
```


```python
def _img_score_top_5(img_path):
    width = 256
    height = 128
    img_size = width * height * 3
    img = mpimg.imread(img_path)
    img = scipy.misc.imresize(img,(height,width))
    
    img_array = img.ravel()
    unique, counts = np.unique(img_array, return_counts=True)
    top_5 = sorted(counts,reverse = True)[0:5]
    top_5_score = 1 - (np.array(top_5).sum())/img_size
    
    gray_img_size = width * height
    gray_img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    gray_img_array = gray_img.ravel()
    unique, counts = np.unique(gray_img_array, return_counts=True)
    gray_top_5 = sorted(counts,reverse = True)[0:5]
    gray_top_5_score = 1 - (np.array(gray_top_5).sum())/gray_img_size
    
    return "{0:.2f}".format(top_5_score)+" {0:.2f}".format(gray_top_5_score)
```


```python
image_count = 12
fig,axs = plt.subplots(ncols = 3,nrows = 3*int(image_count/3),figsize=(20,40))
for i in range(int(image_count/3)):
    for j in range(3):
        imgpath = 'images/show/{}.jpeg'.format(i*3+j+1)
        img=mpimg.imread(imgpath)
        img = scipy.misc.imresize(img,(128,256))
        axs[3*i][j].imshow(img)
        img_array = img.ravel()
        axs[3*i+1][j].hist(img_array, bins=256, range=(0.0, 255.0), fc='k', ec='k')
        axs[3*i+1][j].set_title('score:'+ _img_score_top_5(imgpath))
        
        gray_img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
        gray_img_array = gray_img.ravel()
        axs[3*i+2][j].hist(gray_img_array, bins=256, range=(0.0, 255.0), fc='k', ec='k')
fig.subplots_adjust(hspace=0.3)
plt.show()
```


![png](output_3_0.png)

