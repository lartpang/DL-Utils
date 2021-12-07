# 数据增强相关的代码

## `tv_based`

主要针对`torchvision`中的数据增强类进行了简单的扩展，使其支持包含多个数据的列表作为输入参数。

## `albu_based`

主要针对`albumentation`中的旋转进行了修改，将`mask`的插值方法改为双线性插值。
