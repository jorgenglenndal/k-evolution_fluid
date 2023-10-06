import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import cm

from matplotlib import colors
#from matplotlib.colors import LogNorm
import sys
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import colorcet as cc



H0 = 0.100069 # directly from k-evolution


path = '/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/data/L300_N256_cs2_1e_6/snapshots/'


with open(path + "kess_snapshots.txt", 'r') as file:
    z = []
    for line in file:
        if line.startswith('#'):

            continue
    
        else:
           words = line.split()
           #for word in range(len(words)):
           #    words[word] = float(words[word])
               
           #if words[1] > 99:
           #   print("Blowup before z = 99")
           #   continue
           #else:
           z.append(float(words[1]))

with open(path + "div_variables.txt", 'r') as file:
    z_div_variables = []
    delta_div_variables =[]
    delta_ratio_div_variables = []
    for line in file:
        if line.startswith('#'):
            continue
    
        else:
            words = line.split()
           #for word in range(len(words)):
           #    #words[word] = float(words[word])
           #    z.append(words[word])
            z_div_variables.append(float(words[0]))
            delta_div_variables.append(float(words[1]))
            #delta_ratio_div_variables.append(float(words[2]))


           #if words[1] > 99:
           #   print("Blowup before z = 99")
           #   continue
           #else:
           #   z.append(words[1])
z_div_variables = np.array(z_div_variables)
delta_div_variables = np.array(delta_div_variables)
delta_ratio_div_variables = np.array(delta_ratio_div_variables)
#z_div_variables = abs(z_div_variables)
#delta_div_variables = abs(delta_div_variables)
###plt.scatter(z_div_variables,delta_ratio_div_variables)
###plt.plot(np.ones(2)*z[0],np.linspace(np.nanmin(delta_div_variables[1:])/100,np.nanmax(delta_div_variables[1:])*100,2))
###
###plt.gca().invert_xaxis()
###plt.yscale('log')
###plt.show()

"""
Plotting the largest perturbation vs. redshift
"""
#plt.semilogy(z_div_variables,delta_div_variables)
plt.scatter(z_div_variables[1:],delta_div_variables[1:],color='k')
plt.plot(z_div_variables[1:],delta_div_variables[1:],color ='r')
plt.plot(np.ones(2)*z[1],np.linspace(np.nanmin(delta_div_variables[1:])/100,np.nanmax(delta_div_variables[1:])*100,2))
#plt.plot(z_div_variables[1:],delta_div_variables[1:])
plt.yscale('log')
plt.gca().invert_xaxis()
plt.xlabel("z")
plt.title('The largest |(pi - pi_avg)/pi_avg| at every redshift',size=15)
#plt.title("Largest absolute value of the perturbations in field")
plt.show()

"""
Plotting the Figure 2...
"""
#path_to_cs2_data = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/data/"

cs2 = [1e-10,1e-11,1e-12,1e-5,1e-7,1e-9,1e-13,1e-8,1e-6,5e-6]
zb = [15.9305,33.3975,56.7875,0.524534,4.01823,6.97263,75.2454,4.2553,3.18422,1.32337]

plt.title(r"$N_{\mathrm{grid}} = N_{\mathrm{particles}} = 256^3, L = 300 \mathrm{Mpc/h}, w = -0.9$")
plt.scatter(zb,cs2)
plt.ylabel(r"$c_s^2$")
plt.xlabel(r"$z_b$")
plt.yscale('log')
plt.show()

#for i in range(len(z)):
    #print(z[i])
#sys.exit(0)
filenames = []
#filenames.append('/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test4/pi_k_5.h5')

for i in range(len(z)):
    filenames.append(path + "pi_k_" + str(i+1)+'.h5')

###filenames.append(path + "snap_000_pi_k.h5")
###filenames.append(path + "snap_001_pi_k.h5")
###filenames.append(path + "snap_002_pi_k.h5")

#filenames = ['/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_2.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_3.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_4.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_5.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_6.h5','/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test2/pi_k_7.h5']
#info = '/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test1/snapshots.txt'
#zs = [1.89461,1.89415,1.89392,1.89369,1.89347,1.89324]

data = []
ims = []

for i in range(len(filenames)): # excluding last
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



    if i == 0:
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
    #print("y coordinate = " +str(multi_indices[1]))
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


vmax = np.max(np.array(max_list[15]))
#vmax = 10000
#vmax = 10
vmin = np.min(np.array(min_list[15]))
#vmin = min_list[0]
#vmax = max_list[0]
#vmin = 1e-5
#vmax = 10
##vmin = min(data_.get_array().min() for data_ in data)
##vmax = max(data_.get_array().max() for data_ in data)
#print('max = ' + str(vmax)+ ',   min = ' + str(vmin))
base = 1000
def _forward(x):
    return np.emath.logn(base, x)


def _inverse(x):
    return base**x

#min_perc = np.percentile(data[np.argmin(np.array(min_list))],0)
#max_perc = np.percentile(data[np.argmax(np.array(max_list))],99)

#vmin = min_perc
#vmax = 10
#
#norm = colors.FuncNorm((_forward, _inverse), vmin=vmin, vmax=vmax)
#
norm = colors.LogNorm(vmin=vmin, vmax=vmax,clip=False)

#norm = colors.Normalize(vmin=vmin, vmax=vmax,clip=False)




print('Making artists...')
fig, ax = plt.subplots()
images = []
#plt.imshow(data[-3],cmap = "jet", aspect='auto', origin='lower',extent=[0, 300, 0, 300], interpolation='bicubic')#,animated=True)
#plt.colorbar()
#plt.show()
#
#plt.imshow(data[-2],cmap = "jet", aspect='auto', origin='lower',extent=[0, 300, 0, 300], interpolation='bicubic')#,animated=True)
#plt.colorbar()
#plt.show()
#
#######################################################min_perc = np.percentile(data[:17],1)
#######################################################max_perc = np.percentile(data[:17],100)
#######################################################vmin = min_perc*H0
#######################################################vmax = max_perc*H0
#vmax = max_list[15]
min_list = []
max_list = []
for i in range(10,16):
    min_list.append(np.nanmin(data[i]))
    max_list.append(np.nanmax(data[i][data[i] != np.inf]))

#vmin = np.min(min_list)*H0
#vmax = np.max(max_list)*H0
argmin = np.argmin(min_list)
argmax = np.argmax(max_list)
min_perc = np.percentile(data[argmin+10],0)
max_perc = np.percentile(data[argmax+10],100)
vmin = min_perc*H0
vmax = max_perc*H0

#vmax = data[15]
norm = colors.LogNorm(vmin=vmin, vmax=vmax,clip=False)


#plt.imshow(data[-1],cmap=cc.cm.rainbow4, aspect='auto', origin='lower',extent=[0, 300, 0, 300], interpolation='bicubic')#,norm=norm)#,animated=True)
#plt.colorbar()
#plt.show()

for i in range(10,16):
    ###title = plt.text(0.5, 1.01,"z = " +str(z[i]), horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
   
    print(i)
#    min_perc = np.percentile(data[i],0)
#    max_perc = np.percentile(data[i],100)
#    vmin = min_perc
#    vmax = max_perc
#    #vmax = max_list[15]
#
#    #vmax = data[15]
#    norm = colors.LogNorm(vmin=vmin, vmax=vmax,clip=False)

    #im = ax.imshow(data[i], aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=norm interpolation='bicubic',animated=True)
    plt.title(r"$|H_0\pi|$" " at " "z = " +str(z[i]))
    plt.imshow(data[i]*H0,cmap=cc.cm.rainbow4, aspect='auto', origin='lower',extent=[0, 300, 0, 300], interpolation='none',norm=norm)# set interpolation to bicubic for smooth plots
    #plt.imshow(data[i],cmap="seismic", aspect='auto', origin='lower',extent=[0, 300, 0, 300], interpolation='bicubic',norm=norm)#,
    plt.xlabel("[Mpc/h]")
    plt.ylabel("[Mpc/h]")
    plt.colorbar()
    #plt.savefig("fig1")
    plt.show()
    #ax.set_title(i)
    #if i == 42:
    #    plt.imshow(data[i], aspect='auto', origin='lower',extent=[0, 300, 0, 300],norm=norm, interpolation='none',animated=True)
    #    #plt.colorbar()
    #    plt.show()
    ###images.append([im,title])
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