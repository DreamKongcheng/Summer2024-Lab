
## 普通版本Transformer
我第一版是直接使用了“Attention is All You Need”这篇论文中的Transformer模型，对其做了一些改装就直接用来训练mnist数据集了。经过10个epoch的迭代后能够达到95%左右的正确率。不过我没有加入位置编码，所以可能会有一些问题。

在10个epoch后来到了96的准确率
![](img/report/20240714164221.png)
