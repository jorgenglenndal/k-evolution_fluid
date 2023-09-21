import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm

from matplotlib import colors
#from matplotlib.colors import LogNorm
import sys
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation





path = '/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/test_test_test/'

with open(path + "snapshots.txt", 'r') as file:
    z = []
    for line in file:
        if line.startswith('#'):
            continue
    
        else:
           words = line.split()
           for word in range(len(words)):
               words[word] = float(words[word])
               
           if words[1] > 99:
              print("Blowup before z = 99")
              continue
           else:
              z.append(words[1])

#for i in range(len(z)):
    #print(z[i])
#sys.exit(0)
filenames = []
#filenames.append('/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test4/pi_k_5.h5')

for i in range(len(z)):

    filenames.append(path + "pi_k_" + str(i+1)+'.h5')



#filenames = ['/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_2.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_3.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_4.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_5.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_6.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_7.h5']
#info = '/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test1/snapshots.txt'
#zs = [1.89461,1.89415,1.89392,1.89369,1.89347,1.89324]

data = []
ims = []

for i in range(len(filenames)):
    #print(i)
#cbar = 0
#def update(i):
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
    #data.append(np.log10(x))
    data.append(x)
    print('Done file '+str(i+1) + '/' +str(int(len(filenames))))
    #x_log = np.log10(x)
    #not_first = False
    
    #if  i==0:
    #min_overdensity = np.percentile(x,0)
    #max_overdensity = np.percentile(x,100)
    #   im = ax.imshow(x, cmap='magma', aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=colors.LogNorm(vmin=min_overdensity,vmax=max_overdensity), interpolation='bicubic')#)
       #cbar = fig.colorbar(im, ax=ax, orientation='vertical')
       #im.set_array(x.flatten())
     #  ims.append([im])
     #  continue

       #return im
    
       #not_first = True
    #xx = np.linspace(0,300,64)
    #yy = np.linspace(0,300,64)
    #XX,YY = np.meshgrid(xx,yy)

    
    #pcm = ax[0].pcolor(X, Y, Z,
    #               norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
    #               cmap='PuBu_r', shading='auto')
    #fig.colorbar(pcm, ax=ax[0], extend='max')

    #min_overdensity = np.percentile(x,0)
    #max_overdensity = np.percentile(x,100)
    #offset = colors.TwoSlopeNorm(vmin=min_overdensity,vcenter=np.average(x) ,vmax=max_overdensity)

    #x = np.abs(ds_arr[:,multi_indices[1],:])
    #plt.imshow(np.abs(ds_arr[:,multi_indices[1],:])/np.amax(np.abs(ds_arr[:,multi_indices[1],:])), extent=[0, 200, 0, 200],cmap='viridis', aspect='auto', origin='lower')
    #plt.imshow(x,cmap='magma', aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=LogNorm(vmin=min_overdensity,vmax=max_overdensity), interpolation='bicubic')#vmin=min_overdensity,vmax=max_overdensity)#,norm=LogNorm(),extent=[0, 200, 0, 200])#vmin=min_overdensity,vmax=max_overdensity)


    #im = ax.imshow(x,cmap='magma', aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=colors.LogNorm(vmin=min_overdensity,vmax=max_overdensity), interpolation='bicubic',animated=True,)#vmin=min_overdensity,vmax=max_overdensity)#,norm=LogNorm(),extent=[0, 200, 0, 200])#vmin=min_overdensity,vmax=max_overdensity)
    #ims.append([im])
    #return im
    #plt.contourf(XX,YY,x,cmap='magma', origin='lower',extent=[0, 300, 0, 300])#,norm=colors.LogNorm(vmin=min_overdensity,vmax=max_overdensity))#,animated=True
    #plt.colorbar(cm.ScalarMappable(norm=colors.LogNorm(vmin=min_overdensity,vmax=max_overdensity), cmap='magma'), ax=ax)

    #im = ax.imshow(x, animated=True)
    #ims.append([im])
    #plt.title("Redshift = " + str(z[i]))
    #plt.colorbar(label='Colorbar Label')
    #plt.xlabel('Mpc/h comoving')
    #plt.ylabel('Mpc/h comoving')
    #plt.colorbar(label='Colorbar Label')

#ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                #repeat_delay=5000)
#plt.show()
#ani.save("movie.mp4")

    #plt.savefig('number_'+str(i)+'.pdf')
    #plt.show(block=False)
    #plt.pause(2)
    #plt.close()
    #print('all done')
#fig.colorbar(plot,ax=ax, extend='max')
print('Finding extrema...')
min_list = []
max_list = []
for i in range(len(data)):
    min_list.append(np.nanmin(data[i]))
    max_list.append(np.nanmax(data[i][data[i] != np.inf]))


#vmax = np.max(np.array(max_list))
#vmax = 10000
vmax = max_list[0]#0.008
#vmin = np.min(np.array(min_list))
vmin = min_list[0]#
#vmax = vmin*1e3
#vmin = min(data_.get_array().min() for data_ in data)
#vmax = max(data_.get_array().max() for data_ in data)
print('max = ' + str(vmax)+ ',   min = ' + str(vmin))
base = 100
def _forward(x):
    return np.emath.logn(base, x)


def _inverse(x):
    return base**x

#min_perc = np.percentile(data[10],0)
#max_perc = np.percentile(data[10],100)

#vmin = min_perc
#vmax = max_perc

#norm = colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)

norm = colors.LogNorm(vmin=vmin, vmax=vmax,clip=False)

#norm = colors.Normalize(vmin=vmin, vmax=vmax,clip=False)



print('Making artists...')
fig, ax = plt.subplots()
images = []
for i in range(len(data)):
    title = plt.text(0.5, 1.01,"z = " +str(z[i]), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)

    im = ax.imshow(data[i], aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=norm, interpolation='bicubic',animated=True)
    plt.imshow(data[i], aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=norm, interpolation='bicubic')#,animated=True)
    plt.colorbar()
    plt.show()
    #ax.set_title(i)
    #if i == 42:
    #    plt.imshow(data[i], aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=norm, interpolation='none',animated=True)
    #    #plt.colorbar()
    #    plt.show()
    images.append([im,title])
print('Done making artists.')
#print('171')
#plt.show()

#vmin = min(image.get_array().min() for image in images)
#vmax = max(image.get_array().max() for image in images)
#print('176')
#plt.show()

#vmin = min(data_.get_array().min() for data_ in data)
#vmax = max(data_.get_array().max() for data_ in data)

#norm = colors.LogNorm(vmin=vmin, vmax=vmax)
#print('183')
#plt.show()
#for im in images:
#    im.set_norm(norm)


#fig.colorbar(images[0], ax=ax, orientation='horizontal', fraction=.1)
#print('190')
#plt.show()

#fig.colorbar()
###min_overdensity = np.percentile(data[0],0)
###max_overdensity = np.percentile(data[-1],100)
#im = ax.imshow(data[-1], cmap='magma', aspect='auto', origin='lower',extent=[0, 300, 0, 300],vmin=min_overdensity,vmax=max_overdensity, interpolation='bicubic')#vmin=min_overdensity,vmax=max_overdensity)

#vmin = min(image.get_array().min() for image in images)
#vmax = max(image.get_array().max() for image in images)
#norm = colors.Normalize(vmin=vmin, vmax=vmax)
#for im in images:
#    im.set_norm(norm)

#im = ax.imshow(data[-1], cmap='magma', aspect='auto', origin='lower',extent=[0, 300, 0, 300],vmin=min_overdensity,vmax=max_overdensity, interpolation='bicubic')#vmin=min_overdensity,vmax=max_overdensity)

#cbar = fig.colorbar(im)

#im = ax.imshow(([],[]), cmap='viridis')
#line, = ax.imshow([], lw=2)

#def init():
#    line.set_data([], [])
#    return line,

#cbar = fig.colorbar(im, ax=ax, orientation='vertical')
#def update(frame):
#   number = 100*frame/(len(data)-1)
#   formatted_number = f'{number:.3f}'
#   print(formatted_number+ '%')
#   print(frame)
#   im = ax.imshow(data[frame], aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=norm, interpolation='bicubic',animated=True)#vmin
#   #im.set_array(data[frame])
#   #im.set_norm(norm)
#   #im.set_title("Time = " +str(frame))
#       #cax.set_array(tas[frame,:,:].values.flatten())
#
#
#   #im.set_array(data[frame].flatten())
#   #min_overdensity = np.percentile(data[frame],0)
#   #max_overdensity = np.percentile(data[frame],100)
#   #im = ax.imshow(data[frame], aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=norm, interpolation='bicubic')#vmin=min_overdensity,vmax=max_overdensit
#   #fig.colorbar(im)
#   #im = ax.imshow(data[frame])
#   return [im]
##print(len(data))
#fig,ax = plt.subplots()
#im = ax.imshow(data[0], aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=norm, interpolation='bicubic',animated=True)#vmin
##im = ax.imshow(origin='lower',extent=[0, 300, 0, 300],norm=norm,add_colorbar=True,cmap='coolwarm')#vmin
fig.colorbar(cm.ScalarMappable(norm=norm),ax=ax)
#cbar = fig.colorbar(cax)


#fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)
print('Making animation...')
#img_eff = images[:10]
ani = animation.ArtistAnimation(fig, images)#,interval = 5)
#animation.ArtistAnimation(fig, images)
#ani.save("current_test_custom_from_280.mp4")#,dpi=300)
#ani.save("small_test.mp4")
#print('Done all.')
plt.show()