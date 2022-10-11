import numpy as np
import matplotlib.pyplot as plt
import sys
aerfa = 0.3
beta = 0.7
delta = 10e-7
n = 100
p = 30
#随机生成满秩矩阵(p*n)
def matrix():
    value = np.random.randint(0,100,(p,n))#随机生成p*n的A矩阵
    while( np.linalg.matrix_rank(value)<p ):#保证满秩
        value = np.random.randint(0,2,(p,n))
    return value
#随机生成一维向量(n*1)
def vector():
    value = np.random.rand(n,1) # n*1矩阵
    return value
#生成b向量(n*1)
def b_vector(tmp1,tmp2):
    value = np.dot(tmp1,tmp2)
    return value
#函数值
def f(tmp):
    value = 0
    for i in range(0,n,1):
        value += tmp[i][0]*np.log(tmp[i][0])
    return value
#一阶导数(n*1)
def f1(tmp):
    value = np.zeros( (n,1) )
    for i in range(0,n):
        value[i][0] = 1+np.log(tmp[i][0]) 
    return value
#二阶导数矩阵(n*n)
def f2(tmp):
    value = np.random.randint(0,1,(n,n))#生成0矩阵(n*n)
    for i in range(0,n):
        value[i][i]=1/tmp[i][0]
    return value
#求二阶导数的逆矩阵(n*n)
def inverse_f2(tmp):
    value = f2(tmp)#二阶导数
    value = np.linalg.inv(value)#求逆
    return value
#求方向d_nt^k(n*1)
def d_nt(tmp,tmpA):
    temp1 = np.hstack( (f2(tmp),np.transpose(tmpA)) )#横向合并矩阵
    temp2 = np.hstack( (tmpA,np.zeros((p,p))) )
    temp3 = np.vstack( (temp1,temp2) )#纵向合并矩阵
    temp4 = np.vstack( ( -f1(tmp),np.zeros( (p,1) ) ) )
    temp_value = np.dot(np.linalg.inv(temp3),temp4)
    value = np.zeros((n,1))
    for i in range(0,n,1):
        value[i][0] = temp_value[i][0]
    return value
#求lamba
def lamba2(tmp,tmpA):
    inverse_d_nt = np.transpose(d_nt(tmp,tmpA))
    value1 = np.dot(inverse_d_nt,f2(tmp))#d^T*f_2
    value = np.dot(value1,d_nt(tmp,tmpA))#d^T*f_2*d
    return value
#main函数
def main(script,*argv):
    k = 0
    xk = vector()#随机生成初始点(n*1)
    A = matrix()#随机生成矩阵(n*n)
    b = b_vector(A,xk)#随机生成b向量(n*1)
    x=list()#记录迭代次数
    y=list()#记录函数值
    t=list()#记录tk
    residual = list()
    x.append(k)
    y.append(f(xk))
    while( 1/2*lamba2(xk,A) > delta ):
        tk = 1 #步长
        dk = d_nt(xk,A)#方向
        #print("times:",k,np.dot(A,dk))#验证一下Ad=0
        while( f(xk+tk*dk) > f(xk)-aerfa*tk*lamba2(xk,A) ):
            tk *= beta
        xk += tk*dk
        k += 1
        x.append(k)
        y.append(f(xk))
        t.append(tk)
        residual.append(lamba2(xk,A))
    x.append(k)
    y.append(f(xk))
    residual.append(lamba2(xk,A))
    #绘图
    fig = plt.figure(num=1,figsize=(10,5))
    #第一个
    plt.subplot(121)
    plt.scatter(x=x,y=y,color='r')
    plt.xlabel('迭代次数')
    plt.ylabel('函数值')
    plt.title('等式约束的牛顿方法(可行初始点)')
    #第二个
    plt.subplot(122)
    plt.scatter(x=x[1:len(x)],y=residual,color='g')
    plt.xlabel('迭代次数')
    plt.ylabel('残差')
    plt.yscale('log')
    plt.show()
#入口
if __name__=="__main__":
    main(*sys.argv)