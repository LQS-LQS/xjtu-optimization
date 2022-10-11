import numpy as np
import matplotlib.pyplot as plt
import sys
import math
p = 30
n = 100
aerfa = 0.3   
beta = 0.7
delta = 10e-5
xk = np.random.rand(n,1) #生成xk
A= np.random.uniform(-0.1,0.1,(p,n))
b = np.dot(A,xk) #生成b向量
vk = np.random.rand(p,1) #生成列矩阵
def g(tmp):
    global A,b
    temp_M = np.dot(np.transpose(A),tmp)
    value = 0
    for i in range(0,p,1):
        value += b[i][0]*tmp[i][0]
    for i in range(0,n,1):
        value += math.exp(-1-temp_M[i])
    return value
def g1(tmp):
    global A,b
    temp_M = np.dot(np.transpose(A),tmp)
    value = np.zeros((p,1))
    for i in range(0,p,1):
        value[i][0] = b[i][0]
        for j in range(0,n,1):
            value[i][0] +=  -A[i][j] * math.exp(-1- temp_M[j]  )
    return value
def g2(tmp):
    global A,b
    temp_M = np.dot(np.transpose(A),tmp)
    value = np.zeros( ( p,p) )
    for i in range(0,p,1):
        for k in range(0,p,1):
            value[i][k] = 0
            for j in range(0,n,1):
                value[i][k] += A[i][j]*A[k][j]*math.exp(-1-temp_M[j])
    return value
def d_nt(tmp):
    value = -np.dot( np.linalg.inv(g2(tmp)) , g1(tmp))
    return value
def lamba2(tmp):
    value1 = np.dot( np.transpose(d_nt(tmp)), g2(tmp))
    value = np.dot(value1,d_nt(tmp)) 
    return value
def main(script,*argv):
    global vk
    k = 0
    x=list()#记录迭代次数
    y=list()#记录函数值
    x.append(k)
    y.append(g(vk))
    while( 1/2*lamba2(vk) > delta ):
        tk = 1 #步长
        dk = d_nt(vk)#方向
        #print(vk+tk*dk)
        #print(vk)
        print(g(vk+tk*dk))
        while( g(vk+tk*dk) > g(vk)-aerfa*tk*lamba2(vk) ):
            tk *= beta
        vk += tk*dk
        k += 1
        x.append(k)
        y.append(g(vk))
    x.append(k)
    y.append(g(vk))
    #绘图
    print("times:",k)
    print("value:",g(vk))
    plt.scatter(x=x,y=y,color='r')
    plt.xlabel('迭代次数')
    plt.ylabel('函数值')
    plt.show()
#入口
if __name__=="__main__":
        main(*sys.argv)


