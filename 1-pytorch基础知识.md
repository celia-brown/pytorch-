# 了解常见的深度学习框架

## TensorFlow

由Google开发的一款开源软件库，为深度学习或人工神经网络而设计。允许使用流程图创建神经网络和计算模型，是可用于深度学习的最好维护和最为流行的开源库之一。可以使用TensorBoard进行简单的可视化并查看计算流水线。其灵活的架构允许你轻松部 署在不同类型的设备上。

不利的一面是，TensorFlow没有符号循环，不支持分布式学习。此外，它还不支持 Windows。

面向对象：做工程的小伙伴

## Pytorch

是Meta(前Facebook)的框架，前身是Torch，支持动态图，而且提供了Python接口。是一个以Python优先的 深度学习框架，不仅能够实现强大的GPU加速，同时还支持动态神经网络。

缺点：入门很快、速度有点慢、部署很垃圾、

面向对象：适合做学术的小伙伴用

##  PaddlePaddle

百度推出的深度学习框架，算是国人最火的深度学习框架了。

特点：计算图动态图都支持、有高级API、速度快、部署方便、有专门的平台

面向对象：果没有卡那就非常适合，如果算力不缺，建议先看Pytorch，当然也可以PaddlePaddle。

## Keras

Keras可以当成一种高级API，它的后端可以是Theano和tensorFlow(可以想成把TF的很多打包了)。由于是高级API非常的方便，非常适合科研人员上手。

## ONNX

是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。它使得不同的人工智能框架（如 Pytorch, MXNet）可以采用相同格式存储模型数据并交互。用大白话说就是是一个中间件，比如你Pytorch的模型想 转换别的，就得通过ONNX，现在有的框架可以直接转，但是在没有专门支持的时候，ONNX就非常重要了，万物先 转ONNX，ONNX再转万物。ONNX本身也有自己的模型库以及很多开源的算子，所以用起来门槛不是那么高。

入门推荐：不用刻意学习，用到了再看

> 接下来就是进入正片，学习pytorch的基本知识，方便未来深度学习的数据运用。

# Pytorch

## 什么是pytorch

1. PyTorch是一个开源机器学习和深度学习框架。PyTorch 允许您使用 Python 代码操作和处理数据并编写深度学习算法，能够在强大的GPU加速基础上实现张量和动态神经网络。
2. PyTorch是一个基于 Python 的科学计算包，使用 Tensor 作为其核心数据结构，类似于 Numpy 数组。不同的是，PyTorch 可以用GPU来处理数据，提供许多深度学习的算法。
3. PyTorch提供了完整的使用文档、循序渐进的用户指南，作者亲自维护PyTorch论坛，方便用户交流和解决问题。
4. Meta(前Facebook)人工智能研究院FAIR对PyTorch的推广提供了大力支持。作为当今排名前三的深度学习研究机构，FAIR的支持足以确保PyTorch获得持续开发、更新的保障，不至于像一些个人开发的框架那样昙花一现。 如有需要，我们也可以使用Python软件包(如NumPy、SciPy和Cython)来扩展 PyTorch。

5. 相对于TensorFlow，PyTorch的一大优点是它的图是动态的，而TensorFlow框架是静态图，不利于扩展。同时， PyTorch非常简洁，方便使用。
6.  如果说TensorFlow的设计是“Make it complicated”，Keras的设计是“Make it complicated and hide it”，那么 PyTorch的设计则真正做到了“Keep it simple，stupid”。

## 环境安装

打开anaconda Prompt，创建虚拟环境

~~~
conda create -n dlpy39 python=3.9
~~~

激活虚拟环境

~~~
conda activate dlpy39
~~~

### 补充anaconda使用

创建环境

~~~
conda create -n 环境名
~~~

查看当前conda所有环境

~~~
conda info --envs
conda info -e
conda env list
~~~

删除虚拟环境

~~~
conda remove -n dlpy39 --all
~~~

在环境中安装包

~~~
Conda install 包名称
或者pip install 包名称 -i https://pypi.tuna.tsinghua.edu.cn/simple（清华镜像）
或者pip install 包名称 -i  https://pypi.doubanio.com/simple/ （豆瓣镜像）
~~~

查看环境中现有的包

~~~
Conda list
pip list
~~~

退出当前环境

```text
deactivate 环境名
```

## Pytorch-cpu 版本

安装

~~~
pip install torch==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
~~~

验证

~~~
import torch
torch.__version__
~~~

## 张量的基本使用

英文为Tensor，是机器学习的基本构建模块，是以数字方式表示数据的形式。PyTorch 就是将数据封装成张量 （Tensor）来进行运算的。PyTorch 中的张量就是元素为同一种数据类型的多维数组。在 PyTorch 中，张量以 "类"  的形式封装起来，对张量的一些运算、处理的方法被封装在类中。

> 0维张量：将标量转化为张量得到的就是0维张量
>
> 1维张量：将向量转化为张量得到的就是1维张量
>
> 2维张量：将矩阵转化为张量得到的就是2维张量
>
> 多维张量：将多维数组转化为张量得到的就是多维张量

~~~python
import torch
# 0维张量：标量（scalar)
scalar = torch.tensor(1)
print(scalar.ndim)

# 1维张量：向量（vector)
vector = torch.tensor([1, 2, 3])
print(vector.ndim)

# 2维张量：矩阵（matrix)
matrix = torch.tensor([[7, 8], [9, 10]])
print(matrix.ndim)

# 3维张量：张量（tensor)
tensor = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(tensor.ndim)
~~~

## 张量的创建

### 张量基本创建

1. torch.tensor 根据已有**数据**创建张量
2. torch.Tensor 根据**形状**创建张量, 其也可用来创建指定数据的张量
3. torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建**指定类型**的张量

~~~python
import torch
import numpy as np

def test01():
    data = torch.tensor(10)
    print(data)

    data = np.random.randn(2,3)
    data = torch.tensor(data)
    print(data)

    data = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
    data = torch.tensor(data)
    print(data)
def test02():
# 2.1 创建2行3列的张量, 默认 dtype 为 float32
    data = torch.Tensor(2,3)
    print(data)
    # 2.2 注意: 如果传递列表, 则创建包含指定元素的张量
    data = torch.Tensor([10])
    print(data)
    data = torch.Tensor([10, 20])
    print(data)
def test03():
    # 3.1 创建2行3列, dtype 为 int32 的张量
    data = torch.IntTensor(2,3)
    print(data)
    # 3.2 注意: 如果传递的元素类型不正确, 则会进行类型转换
    data = torch.IntTensor([2.5,3.3])
    print(data)
    # 3.3 其他的类型
    data = torch.ShortTensor() # int16
    data = torch.LongTensor()  # int64
    data = torch.FloatTensor() # float32
    data = torch.DoubleTensor() # float64

if __name__ == '__main__':
    # test01()
    # test02()
    test03()
~~~

