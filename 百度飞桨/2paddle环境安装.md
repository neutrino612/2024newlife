https://www.paddlepaddle.org.cn/  官网安装

# 1 conda 安装
https://www.paddlepaddle.org.cn/documentation/docs/zh/install/conda/windows-conda.html#anchor-0

Anaconda是一个免费开源的 Python 和 R 语言的发行版本，用于计算科学，Anaconda 致力于简化包管理和部署。Anaconda 的包使用软件包管理系统 Conda 进行管理。Conda 是一个开源包管理系统和环境管理系统，可在 Windows、macOS 和 Linux 上运行。

（1） 在Anaconda Powershell Prompt 中创建虚拟环境
    conda create -n paddle_24 python=3.9
    3.9为想要安装的Python的版本，安装完毕后，就可以激活paddle环境
（2）激活paddle的环境
    conda activate paddle_24
（3）检查系统的环境
    确认 Python 和 pip 是 64bit，并且处理器架构是 x86_64（或称作 x64、Intel 64、AMD64）架构。下面的第一行输出的是”64bit”，第二行输出的是”x86_64（或 x64、AMD64）”即可：
（4）安装paddle
conda install paddlepaddle==2.6.0 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/

# https://blog.csdn.net/maxle/article/details/121763416  机器学习“CUDA”、“飞桨AI Studio”、“PyCharm”、 “Python”、“ Anaconda”、“numpy”、“飞桨PaddlePaddle”辨析

## CPU

计算机的计算处理一般依赖于CPU   也就是我们常说的中央处理器
CPU遵循的是冯诺依曼架构，核心就是存储程序，顺序执行，所以在大规模并行计算能力上极受限制，而更擅长逻辑控制
例如：

编写的代码在编译的时候，计算机首先将硬盘的data 调入内存，以便和cpu配合进行运算
在运算的时候，cpu读取指令，读到的是代码编译后的二进制指令，但在处理指令的时候，依然是一条一条的进行处理

这种结构决定了cpu只能提高读取速度来提高运算效率

## GPU

GPU的并行计算能力，CUDA是英伟达推出的只能用于自家GPU的并行计算框架，只有安装这个框架才能进行复杂的并行计算，主流的深度学习框架也都是基于CUDA进行GPU并行加速的，几乎无一例外

还有一个叫cudnn，是针对深度卷积神经网络的加速库

如果要使用GPU训练模型，就需要安装CUDA和cudnn指令集

## pycharm

pycharm是一个IDE，但它和其他的编程IDE  visual studio相比，最重要的是没有集成编译器

只有代码编辑，项目管理等功能，使用pycharm新建项目首先要指定解释器，可以使用不同的python解释器

安装pycharm  还需要安装python编译器

## python


python是一种跨平台的计算机程序语言，高层次的结合了解释性、编译性、互动性和面向对象的脚本语言。

python解释器也是可以运行py文件的

IDLE是python原生自带的开发环境，是迷你版的IDE，与以上方式不同的是它带有图形界面，有简单的编辑和调试功能，但操作起来比较麻烦

## Anaconda

开源的python发行版本，其包含了conda,python 等180多个科学包和依赖项

相当于把python  numpy pandas scipy 等常用的库自动安装好了
