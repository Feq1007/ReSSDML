# ReSSDML:基于度量空间的数据流可靠性半监督学习

本次工作主要由两部分组成：
1. 度量学习模块：数据在原始空间并不一定能满足类间近，类内远的特点，而度量学习通过构造三元组来训练一个度量网络，使得通过映射后的新数据具有类间近类内远的特性。常见的基于Triplets的度量学习在简单的数据上已经能有较好的表现，但是一是对于某些高维数据表现的不够好，二是希望我们度量学习的结果能与后面数据流模块相结合。所以基于SoftTripel的思想，我们首先用基于同步的方法找到多个簇类中心，然后再通过学习将最数据映射到最近的同类微簇，尽可能的使映射不要那么分散。
2. 数据流在线维护模块：数据流在线维护是基于微簇的。
