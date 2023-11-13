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



H0 = 0.100069 # Gevolution units

with h5py.File("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/snap_001_delta_rho_fluid.h5", "r") as f:
        a_group_key = list(f.keys())[0]
        delta_rho_fluid = f[a_group_key][()] 

with h5py.File("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/friday/snap_000_v_fluid.h5", "r") as f:
        a_group_key = list(f.keys())[0]
        v_fluid = f[a_group_key][()]  # returns as a numpy array
with h5py.File("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/friday/snap_000_delta_rho_fluid.h5", "r") as f:
        a_group_key = list(f.keys())[0]
        delta_rho_fluid = f[a_group_key][()]  
    
#print(np.shape(v_fluid[:,:,:,1]))
#sys.exit(0)

#with h5py.File("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/snap_001_v_x_fluid.h5", "r") as f:
#        a_group_key = list(f.keys())[0]
#        v_x_fluid = f[a_group_key][()] 
#with h5py.File("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/snap_001_v_y_fluid.h5", "r") as f:
#        a_group_key = list(f.keys())[0]
#        v_y_fluid = f[a_group_key][()] 
#with h5py.File("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/snap_001_v_z_fluid.h5", "r") as f:
#        a_group_key = list(f.keys())[0]
#        v_z_fluid = f[a_group_key][()] 

#print(np.unravel_index(np.argmax(ds_arr), ds_arr.shape)[1])
#print(np.shape(ds_arr))
#index = np.argmax(np.ndarray.flatten(abs()))
#multi_indices = np.unravel_index(index, second_derivative_y.shape)
#x = np.abs(ds_arr[:,multi_indices[1],:])
#plt.imshow(ds_arr[:,np.unravel_index(np.argmax(ds_arr), ds_arr.shape)[1],:])
#plt.show()
from mayavi import mlab
#print(np.min(delta_rho_fluid))
#print(np.max(delta_rho_fluid))
#sys.exit(0)
#mlab.clf()
#mlab.points3d(ds_arr)
#mlab.show()

#print(np.shape(ds_arr)[1])
#sys.exit(0)
###vmin = np.percentile(delta_rho_fluid,0.1)
###vmax = np.percentile(delta_rho_fluid,99.9)
###mlab.volume_slice(delta_rho_fluid,vmin=vmin,vmax=vmax)
####mlab.contour3d(np.linspace(0,1,np.shape(ds_arr)[0]),np.linspace(0,1,np.shape(ds_arr)[0]),np.linspace(0,1,np.shape(ds_arr)[0]),ds_arr,vmin=vmin,vmax=vmax)
###mlab.show()
v_speed_fluid = np.sqrt(v_fluid[:,:,:,0]**2+ v_fluid[:,:,:,1]**2 +v_fluid[:,:,:,2]**2)
avg_v_fluid = np.average(v_speed_fluid)

#sys.exit(0)
#delta_v_fluid = v_speed_fluid - avg_v_fluid
#over_velocity_v_fluid = delta_v_fluid/avg_v_fluid
over_velocity_v_fluid = v_speed_fluid/avg_v_fluid - 1.
#print(over_velocity_v_fluid)
#sys.exit(0)
#mlab.volume_slice(over_velocity_v_fluid)
n_part = np.shape(v_fluid)[0]
#print(n_part)
#sys.exit(0)
x = np.linspace(0,1,n_part)
xx,xy,xz = np.meshgrid(x,x,x)

"""
Tried to implement my own mask_points function, but result was not improved compared to pre-existing method
###import random as rd
###import secrets
###random = []
###for i in range(n_part**3):
###   random.append(secrets.randbelow(n_part**3))
###random = np.array(random)
###random_shape = np.zeros((n_part,n_part,n_part))
###random = random.reshape(random_shape.shape)
###percentage = 20 # percentage of total number of points to plot
###condition = random > (n_part**3-1)*(100-percentage)/100
###over_velocity_v_fluid_reduced = over_velocity_v_fluid[condition]
###xx_reduced = xx[condition]
###xy_reduced = xy[condition]
###xz_reduced = xz[condition]
"""

obj = mlab.points3d(xx,xy,xz,delta_rho_fluid,mask_points = 20,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=0.5,scale_factor=1/n_part*1.5,colormap = 'coolwarm')
#obj = mlab.points3d(xx_reduced,xy_reduced,xz_reduced,over_velocity_v_fluid_reduced,mask_points = 1,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=0.5,colormap = 'coolwarm',scale_factor=1/128*1.5)
mlab.colorbar(orientation='vertical')#,title= "overdensity for velocity")
obj.glyph.glyph.clamping = True
mlab.outline(obj)
#mlab.xlabel('300 [Mpc/h]')
#mlab.ylabel('300 [Mpc/h]')
#mlab.zlabel('300 [Mpc/h]')




mlab.show()
#mlab.clf()
mlab.quiver3d(v_fluid[:,:,:,0], v_fluid[:,:,:,1], v_fluid[:,:,:,2],mask_points = 10,opacity = 0.3,mode = "2darrow",line_width=0.8)
mlab.show()



#path = '/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/data2/L300_N256_cs2_1e_11/'
#path2 = '/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/data/'
#path_not_source = '/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/L300_N512/kess_not_source_gravity/'
#path_source = '/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/data3/'


#with open(path + "kess_snapshots.txt", 'r') as file:
#    z = []
#    for line in file:
#        if line.startswith('#'):
#
#            continue
#    
#        else:
#           words = line.split()
#           #for word in range(len(words)):
#           #    words[word] = float(words[word])
#               
#           #if words[1] > 99:
#           #   print("Blowup before z = 99")
#           #   continue
#           #else:
#           z.append(float(words[1]))

import os
def blowup_redshifts(dir_path):
    test_list = []
    for path, dirnames, filenames in os.walk(dir_path):
        test_list.append(dirnames)
    subdirs = test_list[0]
    for i in range(len(subdirs)):
        subdirs[i] = dir_path + subdirs[i] + "/"
    #print(subdirs)
    #sys.exit(0)

    blowup_redshift = []
    cs2_kessence = []
    for i in range(len(subdirs)):
        #print(i)
        if len(cs2_kessence) > len(blowup_redshift): # if no blowup, we do not want sound speed
            cs2_kessence.pop()
            #print(str(i)+ " cs2 popped")
            #print("popped")
        with open(subdirs[i] + "div_variables.txt", 'r') as file:
            for line in file:
                if line.startswith("#"):
                    continue
                elif line.startswith("cs2_kessence"):
                    words = line.split()
                    cs2_kessence.append(float(words[1]))
                    #print(cs2_kessence[-1])
                    #print(str(i)+ " cs2 added")
                    continue
                else:
                    words = line.split()
                    if "inf" in words[1] or "nan" in words[1] or abs(float(words[1])) > 1:
                        blowup_redshift.append(float(words[0]))
                        #print(str(i) + " blowup")
                        #print(cs2_kessence[-1])
                        break
                    #if abs(float(words[1])) > 1000:
                    #    blowup_redshift.append(float(words[0]))
                    #    break
    if len(cs2_kessence) > len(blowup_redshift): # if no blowup, we do not want sound speed
            cs2_kessence.pop()
            #print(str(len(subdirs)-1)+ " cs2 popped")    
    return blowup_redshift,cs2_kessence
#print(blowup_redshift)
#print(cs2_kessence)

""" plotting z vs. avg_T00_Kess """
def z_vs_avg_T00_Kess(path):
    with open(path,"r") as file:
        z = []
        var1 = [] #avg_T00_Kess = []
        var2 = [] #avg_T00_Kess_plus_rho_smg = []
        #var3 = [] #rho_smg = []
        for line in file:
            if line.startswith("#"):
                continue
            else:
                words = line.split()
                #if float(words[1]) > 1e100 or float(words[1]) < 1e-13:
                #    continue
                
                z.append(float(words[0]))
                var1.append(float(words[1]))
                var2.append(float(words[2]))
                #rho_smg.append(float(words[3]))
    return np.array(z), np.array(var1), np.array(var2)


"""plot with wrong T00 expression in gevolution"""
###path_lin = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/energy_overdensity/old/linear/avg_T00_Kess.txt"
###path_nonlin = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/energy_overdensity/old/nonlinear/avg_T00_Kess.txt"
###z_lin,var1_lin,var2_lin = z_vs_avg_T00_Kess(path_lin)
###z_nonlin,var1_nonlin,var2_nonlin = z_vs_avg_T00_Kess(path_nonlin)
###
###plt.plot(z_nonlin,abs(var1_nonlin),color="r")
###plt.plot(z_nonlin,abs(var2_nonlin),"--",color='r')
###plt.yscale('log')
###plt.gca().invert_xaxis()
####plt.show()
####sys.exit(0)



path_lin = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/energy_overdensity/linear/avg_T00_Kess.txt"
path_nonlin = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/energy_overdensity/nonlinear/avg_T00_Kess.txt"
z_lin,var1_lin,var2_lin = z_vs_avg_T00_Kess(path_lin)
z_nonlin,var1_nonlin,var2_nonlin = z_vs_avg_T00_Kess(path_nonlin)
#print(z_nonlin)
#plt.plot(z_nonlin[:-1],abs(var1_nonlin[:-1]),color='k')
plt.plot(z_nonlin[:-1],abs(var2_nonlin[:-1]),"--",color='k')
plt.yscale('log')
plt.gca().invert_xaxis()
plt.show()
sys.exit(0)
"""
fig,ax = plt.subplots(1,2)
fig.suptitle("Average dark energy fluid energy-overdensity," r"$\frac{\delta \rho}{\rho_\mathrm{CLASS}}$" ", using background value from CLASS as average",size=(25))

ax[0].semilogy(z_lin,)
ax[0].set_title('Linear dark energy')
#plt.ylim(1e-10,np.max(np.array(T00_test)))
#plt.plot(z_test,T00_test)

ax[0].set_xlabel("z")
ax[0].invert_xaxis()

ax[1].semilogy(z_nonlin,T00_nonlin)
ax[1].set_title('Non-linear dark energy')
#plt.ylim(1e-10,np.max(np.array(T00_test)))
#plt.plot(z_test,T00_test)
#plt.gca().invert_xaxis()
ax[1].set_xlabel("z")
ax[1].invert_xaxis()
plt.show()
fig,ax1 = plt.subplots()
ax1.plot(z_nonlin,(avg_tot_T00_Kess_nonlin-rho_smg_nonlin)/rho_smg_nonlin,label=  r'$\frac{<\rho_\mathrm{CLASS} + \delta \rho_\mathrm{non-linear} > - \rho_\mathrm{CLASS}}{\rho_\mathrm{CLASS}} $',color='k')
ax1.set_yscale('log')
ax1.set_ylabel("Non-linear",size=15)
ax1.invert_xaxis()
ax2 = ax1.twinx()
ax2.set_ylabel('Linear',color='r',size=15)

ax2.plot(z_lin,(avg_tot_T00_Kess_lin-rho_smg_lin)/rho_smg_lin,label=r'$\frac{<\rho_\mathrm{CLASS} + \delta \rho_\mathrm{linear} > - \rho_\mathrm{CLASS}}{\rho_\mathrm{CLASS}} $',color='r')
ax2.set_yscale('linear')
#ax2.invert_xaxis()
#plt.plot(z_lin,rho_smg_lin,label='rho_smg Linear',marker='1')
#plt.scatter(z_nonlin,rho_smg_nonlin,label='Background dark energy density CLASS',color='k')
ax1.legend(bbox_to_anchor=(0, 1), loc='upper left',fontsize=25)
ax2.legend(bbox_to_anchor=(0.9, 1), loc='upper right',fontsize=25)
#plt.gca().invert_xaxis()
ax1.set_xlabel("z")
plt.show()
sys.exit(0)
"""
                     




"""

"""
#Blowup redshift with cs2. Kess sourcing and not sourcing gravity 
"""

path_L300_N256_kess_source_gravity = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/L300_N256/kess_source_gravity/"
path_L300_N256_kess_not_source_gravity = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/L300_N256/kess_not_source_gravity/"

br,cs2 = blowup_redshifts(path_L300_N256_kess_not_source_gravity)
br_s,cs2_s = blowup_redshifts(path_L300_N256_kess_source_gravity)

"""
#not source gravity blows up for cs^2 = 1.3e-5 and not for 1.4e-5. Same for source gravity.
"""
plt.scatter(br,cs2,label='Not sourcing gravity',marker = 'x',s=100)
plt.scatter(br_s,cs2_s,label='Sourcing gravity')
#plt.show()
plt.legend()
plt.title(r"$N_{\mathrm{grid}} = N_{\mathrm{particles}} = 256^3, L = 300 \mathrm{Mpc/h}, w = -0.9$")
#plt.scatter(zb,cs2)
plt.ylabel(r"$c_s^2$")
plt.xlabel(r"$z_b$")
plt.yscale('log')
####plt.show()
#sys.exit(0)
#print("96")
path_L300_N128_kess_source_gravity = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/L300_N128/kess_source_gravity/"
path_L300_N128_kess_not_source_gravity = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/L300_N128/kess_not_source_gravity/"

br,cs2 = blowup_redshifts(path_L300_N128_kess_not_source_gravity)
br_s,cs2_s = blowup_redshifts(path_L300_N128_kess_source_gravity)
#print(cs2)
#print(br)
"""
#not source gravity 8e-6 does not blow up, 7e-6 blows up. Same for source gravity
"""

plt.scatter(br,cs2,label='Not sourcing gravity',marker = 'x',s=100)
plt.scatter(br_s,cs2_s,label='Sourcing gravity')
#plt.show()
plt.legend()
plt.title(r"$N_{\mathrm{grid}} = N_{\mathrm{particles}} = 128^3, L = 300 \mathrm{Mpc/h}, w = -0.9$")
#plt.scatter(zb,cs2)
plt.ylabel(r"$c_s^2$")
plt.xlabel(r"$z_b$")
plt.yscale('log')
###plt.show()
#sys.exit(0)


path_L300_N512_kess_source_gravity = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/L300_N512/kess_source_gravity/"
path_L300_N512_kess_not_source_gravity = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/L300_N512/kess_not_source_gravity/"

br,cs2 = blowup_redshifts(path_L300_N512_kess_not_source_gravity)
br_s,cs2_s = blowup_redshifts(path_L300_N512_kess_source_gravity)

"""
#not source gravity blows up for cs^2 = 2e-5 and not for 2.5e-5. Same for source gravity.
"""

plt.scatter(br,cs2,label='Not sourcing gravity',marker = 'x',s=100)
plt.scatter(br_s,cs2_s,label='Sourcing gravity')
#plt.show()
plt.legend()
plt.title(r"$N_{\mathrm{grid}} = N_{\mathrm{particles}} = 512^3, L = 300 \mathrm{Mpc/h}, w = -0.9$")
#plt.scatter(zb,cs2)
plt.ylabel(r"$c_s^2$")
plt.xlabel(r"$z_b$")
plt.yscale('log')
###plt.show()
#sys.exit(0)
"""



# density perturbation from k-evolution
with h5py.File("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_rho/snap_000_T00_kess.h5", "r") as f:
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
#x = np.abs(ds_arr[:,multi_indices[1],:])
plt.imshow(ds_arr[:,multi_indices[1],:])
plt.show()



#with open(path + "div_variables.txt", 'r') as file:
#    z_div_variables = []
#    H0_pi_div_variables = []
#    delta_div_variables =[]
#    delta_ratio_div_variables = []
#    for line in file:
#        if line.startswith('#'):
#            continue
#    
#        else:
#            words = line.split()
#           #for word in range(len(words)):
#           #    #words[word] = float(words[word])
#           #    z.append(words[word])
#            if abs(float(words[1])) > 1000:
#                
#            z_div_variables.append(float(words[0]))
#            #delta_div_variables.append(float(words[1]))
#            #delta_ratio_div_variables.append(float(words[2]))


           #if words[1] > 99:
           #   print("Blowup before z = 99")
           #   continue
           #else:
           #   z.append(words[1])

###with open(path + "div_variables.txt", 'r') as file:
###    z_div_variables = []
###    H0_pi_div_variables = []
###    delta_div_variables =[]
###    delta_ratio_div_variables = []
###    for line in file:
###        if line.startswith('#'):
###            continue
###    
###        else:
###            words = line.split()
###           #for word in range(len(words)):
###           #    #words[word] = float(words[word])
###           #    z.append(words[word])
###            if abs(float(words[1])) > 1000:
###                
###            z_div_variables.append(float(words[0]))
###            #delta_div_variables.append(float(words[1]))
###            #delta_ratio_div_variables.append(float(words[2]))
###
###
###           #if words[1] > 99:
###           #   print("Blowup before z = 99")
###           #   continue
###           #else:
###           #   z.append(words[1])
###z_div_variables = np.array(z_div_variables)
###delta_div_variables = np.array(delta_div_variables)
###delta_ratio_div_variables = np.array(delta_ratio_div_variables)
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
###plt.scatter(z_div_variables[1:],delta_div_variables[1:],color='k')
###plt.plot(z_div_variables[1:],delta_div_variables[1:],color ='r')
###plt.plot(np.ones(2)*z[1],np.linspace(np.nanmin(delta_div_variables[1:])/100,np.nanmax(delta_div_variables[1:])*100,2))
####plt.plot(z_div_variables[1:],delta_div_variables[1:])
###plt.yscale('log')
###plt.gca().invert_xaxis()
###plt.xlabel("z")
###plt.title('The largest |(pi - pi_avg)/pi_avg| at every redshift',size=15)
####plt.title("Largest absolute value of the perturbations in field")
###plt.show()

"""
Plotting the Figure 2...
"""
#path_to_cs2_data = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/test_cs2/data/"
#4.04238
#cs2 = [1e-10,1e-11,1e-12,1e-5,1e-7,1e-9,1e-13,1e-8,1e-6,5e-6]
#zb = [15.9305,33.3975,56.7875,0.524534,4.01823,6.97263,75.2454,4.2553,3.18422,1.32337]

#plt.title(r"$N_{\mathrm{grid}} = N_{\mathrm{particles}} = 256^3, L = 300 \mathrm{Mpc/h}, w = -0.9$")
##plt.scatter(zb,cs2)
#plt.ylabel(r"$c_s^2$")
#plt.xlabel(r"$z_b$")
#plt.yscale('log')
#plt.show()

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
for i in range(50):
    min_list.append(np.nanmin(data[i]))
    max_list.append(np.nanmax(data[i][data[i] != np.inf]))

#vmin = np.min(min_list)*H0
#vmax = np.max(max_list)*H0
argmin = np.argmin(min_list)
argmax = np.argmax(max_list)
min_perc = np.percentile(data[argmin],1)
max_perc = np.percentile(data[argmax],100)
vmin = min_perc*H0
vmax = max_perc*H0

#vmax = data[15]
norm = colors.LogNorm(vmin=vmin, vmax=vmax,clip=False)


#plt.imshow(data[-1],cmap=cc.cm.rainbow4, aspect='auto', origin='lower',extent=[0, 300, 0, 300], interpolation='bicubic')#,norm=norm)#,animated=True)
#plt.colorbar()
#plt.show()

for i in range(30,len(data)):
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
    plt.imshow(data[i]*H0,cmap=cc.cm.rainbow4, aspect='auto', origin='lower',extent=[0, 300, 0, 300], interpolation='bicubic',norm=norm)# set interpolation to bicubic for smooth plots
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