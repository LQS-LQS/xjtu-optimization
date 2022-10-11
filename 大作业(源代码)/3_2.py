import math
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
aerfa = 0.3
beta = 0.7
delta = 10e-7
n = 100
p = 50
#随机生成满秩矩阵(p*n)
def matrix():
    value = np.random.randint(0,100,(p,n))#随机生成p*n的A矩阵
    while( np.linalg.matrix_rank(value)<p ):#保证满秩
        value = np.random.randint(0,100,(p,n))
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
    for i in range(0,n,1):
        value[i][0] = 1+np.log(tmp[i][0]) 
    return value
#二阶导数矩阵(n*n)
def f2(tmp):
    value = np.zeros( (n,n) )#生成0矩阵(n*n)
    for i in range(0,n,1):
        value[i][i]=1/tmp[i][0]
    return value
#r的二范数
def norm_r2(tmpx,tmpv,tmpA,tmpb):
    value = 0
    value1 = f1(tmpx)+np.dot(np.transpose(tmpA),tmpv)
    value2 = np.dot(tmpA,tmpx) - tmpb
    for i in range(0,n,1):
        value += value1[i][0]*value1[i][0]
    for i in range(0,p,1):
        value += value2[i][0]*value2[i][0]
    return math.sqrt(value)
def get_norm(tmpx,tmpv,tmpA,tmpb):
    value_1 = 0
    value_2 = 0
    value1 = f1(tmpx)+np.dot(np.transpose(tmpA),tmpv)
    value2 = np.dot(tmpA,tmpx) - tmpb
    for i in range(0,n,1):
        value_1 += value1[i][0]*value1[i][0]
    for i in range(0,p,1):
        value_2 += value2[i][0]*value2[i][0]
    return {math.sqrt(value_1),math.sqrt(value_2)}
#xk和vk的方向
def one_d_nt(tmpx,tmpA,tmpv,tmpb,flag):
    temp1 = np.hstack( (f2(tmpx),np.transpose(tmpA)) )#横向合并矩阵
    temp2 = np.hstack( (tmpA,np.zeros((p,p))) )#横向合并矩阵
    temp3 = np.vstack( ( temp1,temp2 ) )#纵向合并矩阵
    temp4 = -np.vstack( (f1(tmpx)+np.dot(np.transpose(tmpA),tmpv),np.dot(tmpA,tmpx)-tmpb) )#纵向合并矩阵
    if np.linalg.matrix_rank(temp3) == n+p:
        temp_value = np.dot(np.linalg.inv(temp3),temp4)
    else:#伪逆矩阵
        temp_value = np.dot(scipy.linalg.pinv(temp3),temp4)
    value_dxk = np.zeros((n,1))
    value_dvk = np.zeros((p,1))
    if flag == True:
        for i in range(0,n,1):
            value_dxk[i][0] = temp_value[i][0]
        return value_dxk
    else:
        for i in range(n,n+p,1):
            value_dvk[i-n][0] = temp_value[i][0]
        return value_dvk
#检查Ax=b
def check(tmp1,tmp2):
    for i in range(0,p,1):
        if abs(tmp1[i][0]-tmp2[i][0]) > delta:
            return False
        return True
#获得合法的tk值
def Get_Valid_tk(tmpx,tmpdx):
    tk = 1
    while(1):
        for i in range(0,n,1):
            if ( tmpx[i][0] + tk*tmpdx[i][0] < 0 ) :
                break
        if (i == n-1):
            return tk
        else:
            tk *= beta
#main函数
def main(script,*argv):
    k = 0
    xk = np.random.rand(n,1)#随机生成初始点(n*1)(不一定满足Axk=b)
    vk = np.random.rand(p,1)#随机生成(p*1)列向量
    A = matrix()#随机生成矩阵(n*n)
    b = np.dot(A,np.random.rand(n,1))#生成b向量(p*1)=(p*n)*(n*1)
    x=list()#记录迭代次数
    y=list()#记录函数值
    t=list()
    residual1=list()
    residual2=list()
    x.append(k)
    y.append(f(xk))
    #不可行初始点迭代到可行点
    while( not( check( np.dot(A,xk),b)==True and norm_r2(xk,vk,A,b)<= delta) ):
        dxk = one_d_nt(xk,A,vk,b,True)#d方向
        dvk = one_d_nt(xk,A,vk,b,False)#v方向
        tk = Get_Valid_tk(xk,dxk)#步长为1时可能会使得x跳出定义域，先确定一个有效的tk值
        while( norm_r2(xk+tk*dxk,vk+tk*dvk,A,b) > (1-aerfa*tk)*norm_r2(xk,vk,A,b) ):
            tk *= beta
        xk += tk*dxk
        vk += tk*dvk
        k += 1
        x.append(k)
        y.append(f(xk))
        t.append(tk)
        v1,v2=get_norm(xk,vk,A,b)
        residual1.append(v1)
        residual2.append(v2)
    #绘图
    fig = plt.figure(num=1,figsize=(10,5))
    #第一个图
    plt.subplot(121)
    plt.scatter(x=x,y=y,color='r')
    plt.xlabel('迭代次数')
    plt.ylabel('函数值')
    plt.title('等式约束的牛顿方法(不可行初始点)')
    #第二个图
    plt.subplot(122)
    plt.scatter(x=x[1:len(x)],y=residual1,color='r',label='对偶残差',s=17)
    plt.scatter(x=x[1:len(x)],y=residual2,color='b',label='原残差',s=17)
    plt.plot(x[1:len(x)],residual1,color='r')
    plt.plot(x[1:len(x)],residual2,color='b')
    plt.yscale('log')#设置纵坐标的缩放，写成m³格式
    plt.legend()
    plt.show()
    print("end")
#入口
if __name__=="__main__":
    main(*sys.argv)