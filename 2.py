import math
import sys
import numpy
import matplotlib.pyplot as plt
best_value = 0
a_array = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]
b_array = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
plot_array_k = []       #记录a=0.3,b=0.7时的迭代
plot_array_value = []   #记录a=0.3,b=0.7时的函数值
plot_array_Totalk = []  #记录迭代次数
plot_array_error = []
delta = 10e-7
#函数值
def f(tmp):
    x1=tmp[0]
    x2=tmp[1]
    value = math.exp(x1+3*x2-0.1) + math.exp(x1-3*x2-0.1) + math.exp(-x1-0.1)
    return value
#一阶导数
def f1(tmp):
    x1=tmp[0]
    x2=tmp[1]
    value1 = math.exp(x1+3*x2-0.1) + math.exp(x1-3*x2-0.1) - math.exp(-x1-0.1)
    value2 = 3*math.exp(x1+3*x2-0.1) - 3*math.exp(x1-3*x2-0.1)
    value=[value1,value2]
    return value
#二范数
def norm(tmp):
    array = f1(tmp)
    value = math.sqrt( pow(array[0],2) + pow(array[1],2) )
    return value
#得到步长
def get_dk(tmp):
    array = f1(tmp)
    value = [ -array[0], -array[1] ]
    return value
#内积
def inner_product(tmp1,tmp2):
    array = f1(tmp1)
    value = array[0]*tmp2[0]+array[1]*tmp2[1]
    return value
#更新点坐标
def update_point(tmp1,int,tmp2):
    value1 = tmp1[0]+int*tmp2[0]
    value2 = tmp1[1]+int*tmp2[1]
    value = [value1,value2]
    return value
#main函数
def get_plot_error():
    for i in range(0,len(plot_array_value),1):
        plot_array_error.append( plot_array_value[i] - best_value )
def main(a,b):
    global plot_array_k,plot_array_value,best_value
    x=[]
    y=[]
    point=[1.0,1.0]
    k = 0
    x.append(k)
    y.append(f(point))
    while( norm(point) > delta):
        tk = 1 #步长
        dk = get_dk(point) #方向
        while( f( [ point[0]+tk*dk[0], point[1]+tk*dk[1] ] ) > f(point) + a*tk*inner_product(point,dk)):
            tk *= b
        point = update_point(point,tk,dk)
        k += 1
        x.append(k)
        y.append(f(point))
    plot_array_Totalk.append(k)
    if a==0.3 and b==0.7:
        best_value = f(point)
        plot_array_k = x
        plot_array_value = y
def plot_figure():
    fig = plt.figure(num=1,dpi=160)
    plt.subplots_adjust(left=0.09, bottom=0.093, right=0.9, top=0.93,hspace=0.4, wspace=0.5)#防止字体标题和xlael交叉
    #第一个图
    plt.subplot(221)
    plt.tick_params(labelsize=7)#刻度减小
    plt.scatter(x = plot_array_k,y = plot_array_value,color='r',s=5)  
    plt.xlabel('迭代次数',fontsize=7)
    plt.ylabel('函数值',fontsize=7)
    plt.title(r'$\alpha$' + '=0.3,'+ r'$\beta$' + '=0.7',fontsize=10)
    #第二个图
    get_plot_error()
    plt.subplot(222)
    plt.tick_params(labelsize=7)
    plt.scatter(x=plot_array_k[1:32],y=plot_array_error[1:32],color='g',s=5)
    plt.xlabel('迭代次数',fontsize=7)
    plt.ylabel('误差',fontsize=7)
    plt.yscale('log')#设置纵坐标的缩放，写成m³格式
    plt.title('误差(忽略初始点)')
    #第三个图
    plt.subplot(223)
    plt.tick_params(labelsize=7)#刻度减小
    plt.xticks([0.1,0.3,0.5,0.7,0.9,1.0])#自设定横坐标
    plt.scatter(x=b_array,y=plot_array_Totalk[0:10],color='b',s=20)
    plt.title(r'$ \alpha =0.2 $'+'时不同'+r'$ \beta $'+'的迭代次数',fontsize=10)
    plt.xlabel(r'$ \beta $',fontsize=7)
    plt.ylabel('迭代次数',fontsize=7)
    #第四个图
    plt.subplot(224)
    plt.tick_params(labelsize=7)#刻度减小
    plt.xticks([0.1,0.2,0.3,0.4,0.5])#自设定横坐标
    plt.scatter(x=a_array,y=plot_array_Totalk[10:19],color='y',s=20)
    plt.title(r'$ \beta =0.6 $'+'时不同'+r'$ \alpha $'+'的迭代次数',fontsize=10)
    plt.xlabel(r'$ \alpha $',fontsize=7)
    plt.ylabel('迭代次数',fontsize=7)
    plt.show()
#程序入口
if __name__=="__main__":
    for i in range(0,len(b_array),1):
        main(0.2,b_array[i])
    for i in range(0,len(a_array),1):
        main(a_array[i],0.6)
    main(0.3,0.7)
    plot_figure()