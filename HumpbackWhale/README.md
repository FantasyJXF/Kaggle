# Humpback Whale Identification



## Evaluation metric

### Mean Average Precision(MAP)

提交根据平均精度Mean Average Precision @ 5 (MAP@5)进行评估:

$$MAP@5=\frac{1}{U}\sum\limits_{u=1}^U\sum\limits_{k=1}^{\min(n,5)}P(k)$$

其中$U$表示图片的数量,$P(K)$表示$k$截至精度,$n$表示每一张图片的预测值.

* 准确率**Precision**

$$P=\frac{\text{#of correct predictions}}{\text{#of all predictions}}=\frac{TP}{TP+FP}$$

* **Precision@k**

$k$截止精度$P(k)$只是通过仅考虑从1级到k级的预测子集来计算精度.

举例说明:

| True |  predicted  |  k   |     P(k)      |
| :--: | :---------: | :--: | :-----------: |
| [x]  | [x,?,?,?,?] |  1   |      1.0      |
| [x]  | [?,x?,?,?]  |  1   |      0.0      |
| [x]  | [?,x,?,?,?] |  2   | $\frac{1}{2}$ |
| [x]  | [?,?,x,?,?] |  2   |      0.0      |
| [x]  | [?,?,x,?,?] |  3   | $\frac{1}{3}$ |

上表中$x$为正确预测,$?$为错误的预测.

通俗的解释是:对于一个样本产生了n个预测值,取前k个预测值,如果在这前k个预测值中有m个为TP真正例,那么Precision@k的值为$\frac{m}{k}$.

* **precision@5 per image**

又根据评价标准所说的"在第一次出现正确的鲸鱼后，计算将停止,即P(1)=1".

意思就是不用累加前5个预测值中的真正例,只需要统计到第一个预测正确的答案即可.在这个比赛中,每幅图只有一个正确答案(TP),因此每幅图的可能的准确度分数要么是0,或者是$P(k)=\frac{1}{k}$.

| true |  predicted  |  k   |       Image score       |
| :--: | :---------: | :--: | :---------------------: |
| [x]  | [x,?,?,?,?] |  1   |           1.0           |
| [x]  | [?,x,?,?,?] |  2   |        0+1/2=0.5        |
| [x]  | [?,?,x,?,?] |  3   |    0/1+0/2+1/3=0.33     |
| [x]  | [?,?,?,x,?] |  4   |  0/1+0/2+0/3+1/4=0.25   |
| [x]  | [?,?,?,?,x] |  5   | 0/1+0/2+0/3+0/4+1/5=0.2 |
| [x]  | [?,?,?,?,?] |  5   | 0/1+0/2+0/3+0/4+1/5=0.0 |
| [x]  | [?,x,?,x,?] |  5   |       0/1+1/2=0.5       |

上表中$x$为正确预测,$?$为错误的预测.

* **Leaderboard score**

最终得分只是图像分数的平均值。

### 代码实现

```python
import numpy as np
import pandas as pd

def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """    
    try:
        # 这里返回的是第一个predictions[idx]与label一致时的idx+1
        # 因为list的序号从0开始
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0

def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l,p in zip(labels, predictions)])
```

```Python
# (true, [predictions])
assert map_per_image('x', []) == 0.0
assert map_per_image('x', ['y']) == 0.0
assert map_per_image('x', ['x']) == 1.0
assert map_per_image('x', ['x', 'y', 'z']) == 1.0
assert map_per_image('x', ['y', 'x']) == 0.5
assert map_per_image('x', ['y', 'x', 'x']) == 0.5
assert map_per_image('x', ['y', 'z']) == 0.0
assert map_per_image('x', ['y', 'z', 'x']) == 1/3
assert map_per_image('x', ['y', 'z', 'a', 'b', 'c']) == 0.0
assert map_per_image('x', ['x', 'z', 'a', 'b', 'c']) == 1.0
assert map_per_image('x', ['y', 'z', 'a', 'b', 'x']) == 1/5
assert map_per_image('x', ['y', 'z', 'a', 'b', 'c', 'x']) == 0.0

assert map_per_set(['x'], [['x', 'y']]) == 1.0
assert map_per_set(['x', 'z'], [['x', 'y'], ['x', 'y']]) == 1/2
assert map_per_set(['x', 'z'], [['x', 'y'], ['x', 'y', 'z']]) == 2/3
assert map_per_set(['x', 'z', 'k'], [['x', 'y'], ['x', 'y', 'z'], ['a', 'b', 'c', 'd', 'e']]) == 4/9
```

```python
train_df = pd.read_csv("../input/train.csv")
train_df.head()
```

```python
labels = train_df['Id'].values
labels
```

```python
# 5 most common Id
# sample_pred = train_df['Id'].value_counts().nlargest(5).index.tolist()
sample_pred = ['new_whale', 'w_23a388d', 'w_9b5109b', 'w_9c506f6', 'w_0369a5c']
predictions = [sample_pred for i in range(len(labels))]
sample_pred
```

```python
map_per_set(labels, predictions)
```

