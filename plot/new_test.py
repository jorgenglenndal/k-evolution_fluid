import numpy as np



#N = 10
#a = np.logspace(-15,1,N,base=np.exp(1))
#c_1 = np.exp(-15.)
#c_2 = 1/(N-1)*np.log(np.exp(1.)/c_1)
#
#a_for_plotting = []
#
#for i in range(N):
#    a_for_plotting.append(c_1*np.exp(i*c_2))
#
#a_for_plotting = np.array(a_for_plotting)
#
#for i in range(N):
#    print(a[i] - a_for_plotting[i])

#print(a)
#print(a_for_plotting)
    


N = 100000
start = -18
stop = 22

x = np.logspace(start,stop,N,base=10)
a = 10**(start)
b = 1/(N-1)*np.log10(10**(stop)/a)

x_test = []

for i in range(N):
    x_test.append(a*10**(i*b))

x_test = np.array(x_test)

for i in range(N):
    print((x_test[i] - x[i])/x[i])

#print(a)
#print(a_for_plotting)