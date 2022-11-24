# -*- coding: utf-8 -*-                                                                                                                        [55/1364]

import numpy as np
import math
import copy
import matplotlib.pyplot as plt

isdebug = True

# 参考文献：机器学习TomM.Mitchell P.137
# 代码参考http://blog.csdn.net/chasdmeng/article/details/38709063

# 指定k个高斯分布参数，这里指定k=2。注意2个高斯分布具有相同均方差Sigma，均值分别为Mu1,Mu2。
def init_data(k,N,**params):
    global X
    global Mu
    global Delta
    global Weight
    global Expectations
    X = np.zeros(0)
    Mu = np.random.random(k)
    Delta = np.random.random(k)*10
    #Weight = np.array([0.6,0.2,0.2])
    Weight = np.random.random(k)/k
    for j in range(k):
        for i in range(0,int(params["w"][j]*N)):
            X = np.append(X, np.random.normal(params["mu"][j], params["delta"][j]))
    Expectations = np.zeros((len(X),k))
    if isdebug:
        print("***********")
        print("初始观测数据X：")
        print(X )
# EM算法：步骤1，计算E[zij]                                                                                                                    [22/1364]
def e_step(k, N):
    global Expectations
    global Mu
    global Delta
    global Weight
    global X
    for i in range(0,len(X)):
        Denom = 0
        Numer = [0.0] * k
        for j in range(0,k):
            Numer[j] = Weight[j]*math.exp((-1/(2*(float(Delta[j]**2))))*(float(X[i]-Mu[j]))**2)/Delta[j]
            Denom += Numer[j]
        for j in range(0,k):
            Expectations[i,j] = Numer[j] / Denom
    if isdebug:
        print("***********")
        print("隐藏变量E（Z）：")
        print(Expectations)

# EM算法：步骤2，求最大化E[zij]的参数Mu
def m_step(k,N):
    global Expectations
    global X
    global Weight
    for j in range(0,k):
        Numer = 0
        Numer_delta = 0
        Denom = 0
        Denum_delta = 0
        for i in range(0,N):
            Numer += Expectations[i,j]*X[i]
            Numer_delta += Expectations[i,j]*(X[i]-Mu[j])**2
            Denom +=Expectations[i,j]
            Denum_delta += Expectations[i,j]
        Mu[j] = Numer / Denom
        Delta[j] = math.sqrt(Numer_delta / Denum_delta)
    Weight[:-1] = (Weight[:-1]+Weight[-1])*Expectations.sum(0)[:-1]/(Expectations.sum(0)[:-1] + Expectations.sum(0)[-1])
    Weight[-1] = 1-Weight[:-1].sum()
    
# 算法迭代iter_num次，或达到精度Epsilon停止迭代
def run(k, N,iter_num,Epsilon, **params):
    init_data(k,N,**params)
    print("初始<u1,u2>, <d1,d2>:", params["mu"], params["delta"])
    for i in range(iter_num):
        Old_Mu = copy.deepcopy(Mu)
        e_step(k,N)
        m_step(k,N)
        print(i,Mu,Delta,Weight)
        if sum(abs(Mu - Old_Mu)) < Epsilon:
            break

if __name__ == '__main__':
    params = {"delta":[6,8, 10],"mu":[40, 20, 10],"w":[0.7,0.1,0.2]}
    iter_num = 2000 # 最大迭代次数
    epsilon = 0.0001    # 当两次误差小于这个时退出
    k = 3
    N = 100000
    run(k,N,iter_num,epsilon, **params)
    #run(mu1,mu2,sigma1, sigma2, w1, w2, k,N,iter_num,epsilon)

    plt.hist(X[:],50)
    plt.savefig("em.png")
