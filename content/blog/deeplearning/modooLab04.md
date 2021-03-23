---
title: '[PyTorch] 모두를 위한 딥러닝 Lab-04'
date: 2021-01-29 23:21:11
category: 'deeplearning'
draft: false
---

# Multivariable Linear regression

다중 선형 회귀라고도 불리는 이것은 복수의 데이터로 $x$가 2개 이상으로 존재하는 경우 이로부터 $y$ 값을 구하는 것을 말한다.

## Theoretical Overview

데이터가 다음과 같이 주어진다고 가정해보자.

| Quiz 1 (x1) | Quiz 2 (x2) | Quiz 3 (x3) | Final (y) |
| :---------: | :---------: | :---------: | :-------: |
|     73      |     80      |     75      |    152    |
|     93      |     88      |     93      |    185    |
|     89      |     91      |     80      |    180    |
|     96      |     98      |     100     |    196    |
|     73      |     66      |     70      |    142    |

이처럼 3개의 x 값으로 y 값을 예측하는 경우

### **Hypothesis Function**의 식은 다음과 같이 세울 수 있다.

---

$$
 H(x) = Wx + b
$$

$$
 H(x_1, x_2, x_3) = x_1w_1 + x_2w_2 + x_3w_3 + b
$$

$$
cost(W, b) = \frac{1}{m} \sum^m_{i=1} \left( H(x^{(i)}) - y^{(i)} \right)^2
$$

- $H(x)$: 주어진 $x$ 값에 대해 예측을 어떻게 할 것인가
- $cost(W, b)$: $H(x)$ 가 $y$ 를 얼마나 잘 예측했는가

---

Cost function은 Simpe Linear Regression과 동일한 MSE 공식을 사용한다.  
(예측 값과 실제 값의 차이의 제곱의 평균)

학습 방식 또한 Optimzer를 설정한 후 Gradient Descent를 사용하는 것으로 동일하게 진행한다.

즉, 최종적으로 달라지는 부분은 Hypothesis 말고는 없다는 것을 알 수 있다.

## Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

```python
# For reproducibility
torch.manual_seed(1)
```

    <torch._C.Generator at 0x7f0b2008b930>

## Naive Data Representation

위의 표에 나온 예시 데이터를 사용해 예측한다.

```python
# 데이터
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```

```python
# 모델 초기화
w1 = torch.zeros(1, requires_grad=True)
w2 = torch.zeros(1, requires_grad=True)
w3 = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w3.item(), w3.item(), b.item(), cost.item()
        ))
```

    Epoch    0/1000 w1: 0.294 w2: 0.297 w3: 0.297 b: 0.003 Cost: 29661.800781
    Epoch  100/1000 w1: 0.674 w2: 0.676 w3: 0.676 b: 0.008 Cost: 1.563634
    Epoch  200/1000 w1: 0.679 w2: 0.677 w3: 0.677 b: 0.008 Cost: 1.497603
    Epoch  300/1000 w1: 0.684 w2: 0.677 w3: 0.677 b: 0.008 Cost: 1.435026
    Epoch  400/1000 w1: 0.689 w2: 0.678 w3: 0.678 b: 0.008 Cost: 1.375730
    Epoch  500/1000 w1: 0.694 w2: 0.678 w3: 0.678 b: 0.009 Cost: 1.319503
    Epoch  600/1000 w1: 0.699 w2: 0.679 w3: 0.679 b: 0.009 Cost: 1.266215
    Epoch  700/1000 w1: 0.704 w2: 0.679 w3: 0.679 b: 0.009 Cost: 1.215693
    Epoch  800/1000 w1: 0.709 w2: 0.679 w3: 0.679 b: 0.009 Cost: 1.167821
    Epoch  900/1000 w1: 0.713 w2: 0.680 w3: 0.680 b: 0.009 Cost: 1.122419
    Epoch 1000/1000 w1: 0.718 w2: 0.680 w3: 0.680 b: 0.009 Cost: 1.079375

## Matrix Data Representation

위의 hypothesis 는 아주 단순하게 정의한 식으로 만일 데이터 $x$의 수가 증가할 수록 식의 길이도 덩달아 길어진다.
그래서 이때 사용하는 방법이 행렬의 곱을 이용하는 방법이다. 행렬의 곱을 사용하면 다음과 같이 식을 간결하게 작성할 수 있다.

$$
\begin{pmatrix}
x_1 & x_2 & x_3
\end{pmatrix}
\cdot
\begin{pmatrix}
w_1 \\
w_2 \\
w_3 \\
\end{pmatrix}
=
\begin{pmatrix}
x_1w_1 + x_2w_2 + x_3w_3
\end{pmatrix}
$$

$$
 H(X) = XW
$$

그리고 이러한 행렬의 곱을 `PyTorch`에서 제공하는 `matmul()` 함수를 사용하면 쉽게 사용이 가능하다.

아래의 코드는

```python
    # H(x) 계산
    hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
```

으로 작성한 코드를 `matmul()` 함수를 사용하는 코드로 수정한 것이다.  
`matmul()` 함수를 사용하면 간결한 것 뿐만 아니라 속도도 더욱 빠르게 계산이 된다.

```python
    # H(x) 계산
    hypothesis = x_train.matmul(W) + b
```

```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```

```python
print(x_train.shape)
print(y_train.shape)
```

    torch.Size([5, 3])
    torch.Size([5, 1])

```python
# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    hypothesis = x_train.matmul(W) + b # or .mm or @

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))
```

    Epoch    0/20 hypothesis: tensor([0., 0., 0., 0., 0.]) Cost: 29661.800781
    Epoch    1/20 hypothesis: tensor([67.2578, 80.8397, 79.6523, 86.7394, 61.6605]) Cost: 9298.520508
    Epoch    2/20 hypothesis: tensor([104.9128, 126.0990, 124.2466, 135.3015,  96.1821]) Cost: 2915.713135
    Epoch    3/20 hypothesis: tensor([125.9942, 151.4381, 149.2133, 162.4896, 115.5097]) Cost: 915.040527
    Epoch    4/20 hypothesis: tensor([137.7968, 165.6247, 163.1911, 177.7112, 126.3307]) Cost: 287.936005
    Epoch    5/20 hypothesis: tensor([144.4044, 173.5674, 171.0168, 186.2332, 132.3891]) Cost: 91.371017
    Epoch    6/20 hypothesis: tensor([148.1035, 178.0144, 175.3980, 191.0042, 135.7812]) Cost: 29.758139
    Epoch    7/20 hypothesis: tensor([150.1744, 180.5042, 177.8508, 193.6753, 137.6805]) Cost: 10.445305
    Epoch    8/20 hypothesis: tensor([151.3336, 181.8983, 179.2240, 195.1707, 138.7440]) Cost: 4.391228
    Epoch    9/20 hypothesis: tensor([151.9824, 182.6789, 179.9928, 196.0079, 139.3396]) Cost: 2.493135
    Epoch   10/20 hypothesis: tensor([152.3454, 183.1161, 180.4231, 196.4765, 139.6732]) Cost: 1.897688
    Epoch   11/20 hypothesis: tensor([152.5485, 183.3610, 180.6640, 196.7389, 139.8602]) Cost: 1.710541
    Epoch   12/20 hypothesis: tensor([152.6620, 183.4982, 180.7988, 196.8857, 139.9651]) Cost: 1.651413
    Epoch   13/20 hypothesis: tensor([152.7253, 183.5752, 180.8742, 196.9678, 140.0240]) Cost: 1.632387
    Epoch   14/20 hypothesis: tensor([152.7606, 183.6184, 180.9164, 197.0138, 140.0571]) Cost: 1.625923
    Epoch   15/20 hypothesis: tensor([152.7802, 183.6427, 180.9399, 197.0395, 140.0759]) Cost: 1.623412
    Epoch   16/20 hypothesis: tensor([152.7909, 183.6565, 180.9530, 197.0538, 140.0865]) Cost: 1.622141
    Epoch   17/20 hypothesis: tensor([152.7968, 183.6643, 180.9603, 197.0618, 140.0927]) Cost: 1.621253
    Epoch   18/20 hypothesis: tensor([152.7999, 183.6688, 180.9644, 197.0662, 140.0963]) Cost: 1.620500
    Epoch   19/20 hypothesis: tensor([152.8014, 183.6715, 180.9666, 197.0686, 140.0985]) Cost: 1.619770
    Epoch   20/20 hypothesis: tensor([152.8020, 183.6731, 180.9677, 197.0699, 140.1000]) Cost: 1.619033

## `nn.Module`과 `F.mse_loss` 사용

**Lab 02**에서 사용했던 다음과 같은 모델을 사용했었다.

```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
```

`nn.Module`을 사용하게 되면

- `PyTorch`에서 제공하는 모델을 쉽게 사용할 수 있다.
- 신경망의 층이 깊은 모델을 쉽게 관리할 수 있다.
- 모델의 재사용이 쉽다.

이번 챕터에서 사용된 데이터를 `Linear` 층을 이용해 예측한다면 단순히 3개의 입력 값을 1개의 출력 값으로 바꿔주는 작업을 진행하면 된다.

그리고 다음은 `PyTorch`에서 제공하는 Cost function을 사용하는 것이다. 이번 과정에서는 `F.mse_loss`가 될 것이다.

`F.mse_loss`를 사용하게 되면

- 직접 식을 세우지 않아도 된다.
- 다른 Cost function으로 변경할 때 매우 간편하다.
- 내부 동작에서의 에러를 생각하지 않아도 돼 디버깅이 간편하다.

```python
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)
```

```python
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 모델 초기화
model = MultivariateLinearRegressionModel()
# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, nb_epochs, cost.item()
    ))
```

    Epoch    0/20 Cost: 31667.597656
    Epoch    1/20 Cost: 9926.266602
    Epoch    2/20 Cost: 3111.513672
    Epoch    3/20 Cost: 975.451355
    Epoch    4/20 Cost: 305.908539
    Epoch    5/20 Cost: 96.042488
    Epoch    6/20 Cost: 30.260750
    Epoch    7/20 Cost: 9.641701
    Epoch    8/20 Cost: 3.178671
    Epoch    9/20 Cost: 1.152871
    Epoch   10/20 Cost: 0.517863
    Epoch   11/20 Cost: 0.318801
    Epoch   12/20 Cost: 0.256388
    Epoch   13/20 Cost: 0.236821
    Epoch   14/20 Cost: 0.230660
    Epoch   15/20 Cost: 0.228719
    Epoch   16/20 Cost: 0.228095
    Epoch   17/20 Cost: 0.227880
    Epoch   18/20 Cost: 0.227799
    Epoch   19/20 Cost: 0.227762
    Epoch   20/20 Cost: 0.227732
