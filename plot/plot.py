import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import colors
from matplotlib.colors import LogNorm
import sys

a = 5
for i in range(a):
    print(i)
    if i ==3:
        a = 10

sys.exit(0)


#info
### 1- tau      2- z    3- a     4- zeta_avg     5- avg_pi       6- avg_phi      7- tau/boxsize  8- H_conf/H0    9- snap_count
#  3.00006         99.9101       0.00990981      1.95906e-14             0       2.38228e-22       2.99928         5.69406               2
#  20.2266         1.89461         0.34547       -9.43277e-07    -0.00259686     -0.0130097        20.2258         1.00844               3
#  20.2266         1.89415        0.345524       0.00378031      -0.00259686     -0.0130097        20.2258         1.00838               4
#  20.2266         1.89392        0.345552          1.8174       -0.00259686     -0.0130097        20.2258         1.00835               5
#  20.2266         1.89369        0.345579          186623       -0.00259686     -0.0130097        20.2258         1.00833               6
#  20.2266         1.89347        0.345606       7.73383e+13     -0.00259686     -0.0130097        20.2258          1.0083               7
#  20.2344         1.89324        0.345633       1.08417e+63     6.09617e+09     -0.0130422        20.2337         1.00827               8

with open('/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test4/snapshots.txt', 'r') as file:
    z = []
    for line in file:
        if line.startswith('#'):
            continue
    
        else:
           words = line.split()
           for word in range(len(words)):
               words[word] = float(words[word])
               
           if words[1] > 99:
              continue
           else:
              z.append(words[1])

#for i in range(len(z)):
    #print(z[i])
#sys.exit(0)
filenames = []
for i in range(len(z)):
    filenames.append('/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test4/pi_k_' + str(i+1)+'.h5')



#filenames = ['/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_2.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_3.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_4.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_5.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_6.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_7.h5']
#info = '/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test1/snapshots.txt'
#zs = [1.89461,1.89415,1.89392,1.89369,1.89347,1.89324]
for i in range(len(filenames)):

    with h5py.File(filenames[i], "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        #print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        #print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        #data = list(f[a_group_key])

        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        #data = list(f[a_group_key])
        # preferred methods to get dataset values:
        #ds_obj = f[a_group_key]      # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array
 #   with open(filenames[i], "r") as f:




    grad_x, grad_y, grad_z = np.gradient(ds_arr)

    # Compute the second derivatives by applying the gradient function again
    second_derivative_x = np.gradient(grad_x)[0]
    second_derivative_y = np.gradient(grad_y)[1]
    second_derivative_z = np.gradient(grad_z)[2]
    index = np.argmax(np.ndarray.flatten(abs(second_derivative_y)))
    multi_indices = np.unravel_index(index, second_derivative_y.shape)


    #print(np.shape(ds_arr))
    ##index = np.argmax(np.ndarray.flatten(np.abs(ds_arr)))
    ##multi_indices = np.unravel_index(index, ds_arr.shape)
    #print(multi_indices)
    #deriv = np.gradient(ds_arr)
    #print(np.shape(deriv[1]))
    #derivderiv = np.gradient(deriv[1])
    #max_derivderiv= np.argmax(np.ndarray.flatten(derivderiv[1]))
    #multi_indices = np.unravel_index(max_derivderiv, derivderiv[1].shape)

    #print(multi_indices)
    x = np.abs(ds_arr[:,multi_indices[1],:])
    min_overdensity = np.percentile(x,0)
    max_overdensity = np.percentile(x,100)

    #offset = colors.TwoSlopeNorm(vmin=min_overdensity,vcenter=np.average(x) ,vmax=max_overdensity)

    #x = np.abs(ds_arr[:,multi_indices[1],:])
    #plt.imshow(np.abs(ds_arr[:,multi_indices[1],:])/np.amax(np.abs(ds_arr[:,multi_indices[1],:])), extent=[0, 200, 0, 200],cmap='viridis', aspect='auto', origin='lower')
    plt.imshow(x,cmap='inferno', aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=LogNorm(vmin=min_overdensity,vmax=max_overdensity), interpolation='bicubic')#vmin=min_overdensity,vmax=max_overdensity)#,norm=LogNorm(),extent=[0, 200, 0, 200])#vmin=min_overdensity,vmax=max_overdensity)
    #plt.title("Redshift = " + str(zs[i]))
    plt.colorbar(label='Colorbar Label')
    plt.xlabel('Mpc/h comoving')
    plt.ylabel('Mpc/h comoving')
    #plt.savefig('number_'+str(i)+'.pdf')
    plt.show(block=False)
    plt.pause(6)
    plt.close()
    #print('all done')
