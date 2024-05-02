from pyclass import *
#from read_files import read_deta_dt
m = 1
s = 1

c = 299792458* m / s
c_km_s = c/1e+3
Gevolution_H0 = 0.100069

#def read_H0(filename):
#    gev  = []
#    kess = []
#    a_gev = []
#    a_kess = []
#    with open(filename, 'r') as infile:
#        for line in infile:
#            if line.startswith("#"):
#                continue
#            else:
#                words = line.split()
#                if words[0] == "gev":
#                    gev.append(float(words[4]))
#                    a_gev.append(float(words[2]))
#                if words[0] == "kess":
#                    kess.append(float(words[4]))
#                    a_kess.append(float(words[2]))
#    #for i in range(a_gev):
#         
#                
#    return gev, kess,a_gev,a_kess
##
#convert_to_cosmic_time_gev, convert_to_cosmic_time_kess,a_gev,a_kess = read_deta_dt("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/hiclass_tests/test1/convert_to_cosmic_velocity.txt")
#
#
#for i in range(len(convert_to_cosmic_time_gev)):
#    convert_to_cosmic_time_gev[i] *= c_km_s*a_gev[i]

root = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/hiclass_tests/test1/"
root = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/demonstrate_blowup/N64/"
root = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fluid/1em3/"

test_file = []
#root = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/"
for i in range(0,20):
#    test_file.append(root + "snap_%03d_pi_k.h5" % i)
    #("tmp_%04d.png" % i)
    #test_file.append(root + "snap_%03d_div_v_upper_fluid.h5" % i)
    #test_file.append(root + "snap_%03d_v_upper_i_fluid.h5" % i)
    #test_file.append(root + "v_upper_i_fluid_" + str(i) +  ".h5")
    #test_file.append(root + "pi_k_" + str(i) +  ".h5")
#for i in range(1,51):
    #("tmp_%04d.png" % i)
    #file.append(root + "snap_%03d_delta_rho_fluid.h5" % i)
    #test_file.append(root + "v_upper_i_fluid_" + str(i) +  ".h5")
    #test_file.append(root + "delta_rho_fluid_" + str(i) +  ".h5")
    test_file.append(root + "snap_%03d_pi_k.h5" % i)
    #test_file.append(root + "snap_%03d_delta_rho_fluid.h5" % i)

    ###test_file.append(root + "snap_%03d_pi_k.h5" % i)                    
#test_file = root + "snap_013_delta_rho_fluid.h5"

 #   test_file.append(root + "snap_00" + str(i)+  "_delta_rho_fluid.h5")
    #test_file = root + "snap_000_pi_k.h5"

#test_file = [root + "snap_000_v_upper_i_fluid.h5", root + "snap_001_v_upper_i_fluid.h5",root + "snap_002_v_upper_i_fluid.h5",root + "snap_003_v_upper_i_fluid.h5"]
#test_file = [root + "snap_000_div_v_upper_fluid.h5", root + "snap_001_div_v_upper_fluid.h5",root + "snap_002_div_v_upper_fluid.h5",root + "snap_003_div_v_upper_fluid.h5"]
#test_file = [root + "snap_001_div_v_upper_fluid.h5"]



#def load_hdf5_data(filename, print_shape = True,print_h5_data_info=True):
#        with h5py.File(filename,'r') as file:
#            keys = list(file.keys())
#            if print_h5_data_info:
#                print(keys)
#                print(type(file[keys[0]]))
#            if len(keys) != 1:
#                print("Wrong format in HDF5 file. File must contain one dataset. Modyfy the 'load_hdf5_data' function. Aborting...")
#                sys.exit(1)
#
#            data = file[keys[0]][()]  # returns as a numpy array. See hdf5 documentation online
#
#            #print(type(file[keys[0]])) 
#        #if self.verbose_bool and print_shape:
#        print("Shape of " + filename +" = " +str(np.shape(data)))
#        return data
#
#data = load_hdf5_data(test_file)
#print(data[:,:,:,2])
#sys.exit(0)

#N = 25
#A = np.zeros((N,N,N))
#for k in range(N):
#    for j in range(N):
#        for i in range(N):
#            A[k,j,i] = i**5-j**5+k**5
#
#np.save("A_test",A)

#file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/demonstrate_blowup/snap_001_pi_k.h5"
#test = visualization_class("A_test.npy",filetype="npy")# 
#test = visualization_class(filename=divergence)# ,indices=["singles",0])
#test = visualization_class(filename=test_file,indices=["range",45,len(test_file),1])#

#test_file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/results/fluid/1em7/snap_000_cdm.h5"
test = visualization_class(filename=test_file,indices=["singles",17])#,12,13,14,15,16])#),len(test_file),1])#

test.rescale_all_data(rescale_all_data_factor=Gevolution_H0)

##test = plot_class(test_file)
#test = plot_class(file,indices=["singles",49])
#test.symmetric_colorbar()
#test.rescale_all_data(rescale_all_data_factor=Gevolution_H0)
#test.rescale_all_data(rescale_all_data_factor=c_km_s)
test.scatter(rescale_factor=1)
#test.log_scale(method="split")
#test.save()
#test.move_camera()
#test.help_indexing()
test.mask(percentile=(10,90,"outside"),method="limits") 
#test.mask(percent=1,method="rng",seed=1111)
#test.offscreen_rendering()
test.show()
#test.move_camera()
test.execute()
#print(test.rescale_data_factor_positive)
print("Done")

sys.exit(0)
import sys
import numpy as np

#a = np.linspace(0,1,11)
#cond = a > 0.5
#b = a[cond]
#b[0] = 100
##print(b[0] == a[0]) 
#print(a)      
##[[0,0,0],[0,0,0],[0,0,0]]
#test = np.array([[[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]],[[2,2,2],[2,2,2],[2,2,2]]],[[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]],[[2,2,2],[2,2,2],[2,2,2]]],[[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]],[[2,2,2],[2,2,2],[2,2,2]]],[[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]],[[2,2,2],[2,2,2],[2,2,2]]]]  )
#test = np.array( [     [    [[0,0],[0,0],[0,0]  ],[ [0,0],[0,0],[0,0]  ]],[ [[0,0],[0,0],[0,0]  ],[ [0,0],[0,0],[0,0]  ]],[ [[0,0],[0,0],[0,0]  ],[ [0,0],[0,0],[0,0]  ]]]   )
test = np.zeros((2,2,2,3))

#np.save("./test_indexing",test)
#test = np.load("./test_indexing.npy")
print(test)
print(np.shape(test))
sys.exit(0)
#
#[[True,True,True],[True,True,True],[True,True,True]]
a = np.array([[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]]])
a[2,2,2] = 100
#a_ = a[]
#a *=-1
#print(np.nanmax(a))
b = np.array([[[False,True,True],[False,True,True],[True,True,True]],[[True,False,True],[True,True,True],[True,True,True]],[[True,True,True],[True,True,True],[True,True,True]]])
c = a[b]
print(np.shape(c))
print(np.nanmax(c))
##print(np.shape(b))
#c = a[b]
#c = c.reshape((3,3,3))
#c[0,0,0] = 100
##print(c)
#print(c[0,0,0] == a[0,0,0])
##print(c)
##print(c[0,0,0] == a[0,0,0])
##print(np.shape(c))
sys.exit(0)

from mayavi import mlab

#x = [1, 2, 3, 4, 5, 6,7,8,9]
n = 5
x = np.linspace(0,1,n)
#x = np.array([-1,-0.5,0,0.5,1,2,4,8,16,32])
y = np.zeros(n)
z = y
#y = [0, 0, 0, 0, 0, 0,0,0,0]
#z = y
#z2 =np.array(y) +1

#s = [1,-3,-2,-1,0,1,2,3,4]
#s = [-25,-12.5,-6.25,0,6.25,12.5,25,50,200]
a = 1
b = 2
s = np.array([0.01,0.1,1,2,4])
#print(s)
#zzz = s.flatten()
#print(zzz)
#sys.exit(0)
#min_ = np.nanmin(s)
#factor = 1/min_
#s2 = np.log10(s*factor)
#print(s2)
s2 = s*2
#sys.exit(0)
#print(x)
#s2 = s*2
# first nd second element should have same size
#x = []
#y = x
#z = x
#s = []
#s2 = np.array(s)/2
#pts = mlab.points3d(x, y, z, s)

#pts2 = mlab.points3d(x, y, z2, s2, scale_factor=2)
#scale_factor_upper = scale_factor if vmax_upper >= vmax_lower else scale_factor*self.vmax_upper/self.vmax
#scale_factor_lower = scale_factor if vmax_lower > vmax_upper else scale_factor*self.vmax_lower/self.vmax
scale_factor = 1/(n-1)/8
scale_factor1 = scale_factor#/np.nanmax(abs(s)) #if np.nanmax(s) >= np.nanmax(s2) else scale_factor*np.nanmax(s)/np.nanmax(s2)
scale_factor2 = scale_factor#/np.nanmax(abs(s2)) #if np.nanmax(s2) >= np.nanmax(s) else scale_factor*np.nanmax(s2)/np.nanmax(s)
pts = mlab.points3d(x, y, z, s,scale_factor=scale_factor1)#*1/1)#,vmin = -0.5,vmax = 1)#scale_factor1)
pts2 = mlab.points3d(x, y, z+0.25, s2, scale_factor=scale_factor2)
pts.glyph.glyph.clamping = False
pts2.glyph.glyph.clamping = False
pts.actor.property.representation = "wireframe"
pts2.actor.property.representation = "wireframe"
#mlab.outline(pts)
mlab.colorbar(object=pts2)
#pts2.glyph.glyph.clamping = False
mlab.show()
sys.exit(0)

#lol = True
#if lol: print('lol'); sys.exit(0)
#print('lol')
#self.random_method = "ccoo"
#if self.random_method == "secrets":
#    print("ok")
#elif self.random_method == "rng": 
#    print("ok")
#else:
#     print("lol")
     #or "secrets" != "rng" : print("lol")

#sys.exit(0)

#import secrets
#rng = np.random.default_rng()
#a = (rng.random(size=10) )
#print(a)
#random = []
#for i in range(10):
#   random.append(secrets.randbelow(10))
#print(random)
#sys.exit(0)



#print(yy)
#sys.exit(0)

#x = np.array([[x,x,x],[x,x,x],[x,x,x]])


"""
#print(yy)
#s = [.5, .6, .7, .8, .9, 1]
#s = np.sqrt(xx**3+yy**2+zz**2)
nx,ny,nz,nv = 2,4,6,3
s = np.zeros((nz,ny,nx,nv))
#print(np.shape(s))
#sys.exit(0)
b = 'b'
arr = np.array([[[b,b,b],[b,b,b],[b,b,b]],[[b,b,b],[b,b,b],[b,b,b]],[[b,b,b],[b,b,b],[b,b,b]]])
print(arr)
sys.exit(0)

"""

#b = [1,2,3]
#a = np.array([[[b,b,b,b],[b,b,b,b],[b,b,b,b],[b,b,b,b]],[[b,b,b,b],[b,b,b,b],[b,b,b,b],[b,b,b,b]],[[b,b,b,b],[b,b,b,b],[b,b,b,b],[b,b,b,b]],[[b,b,b,b],[b,b,b,b],[b,b,b,b],[b,b,b,b]]])
#print(np.shape(a))
#c = a[:,:,:,2]
#print(a)
#sys.exit(0)
"""
"""
#print(np.shape(s))
#sys.exit(0)
#print(s)
#x = [0, 1, 2, 3, 4, 5]
#xx,yy,zz = np.meshgrid(x,x,x,indexing="xy")
xx,yy,zz = np.mgrid[0:1:16j,0:1:16j,0:1:16j]

s = np.zeros((16,16,16)) # zyx
#B = np.zeros((16,16,16))
for k in range(16):
       for j in range(16):
              for i in range(16):
                     #s[i,j,0] = 10
                     s[i,j,k] = np.sqrt(i**2+k**2+j**2)  #str(i) +"_" + str(j) + "_"+str(k)
#s2 = np.zeros((16,16,16))
#for k in range(16):
#       for j in range(16):
#              for i in range(16):
#                     s2[i,j,k] = 5
#                     s2[i,j,0] = 0
#
#
#for z in range(16):
#       for y in range(16):
#              for x in range(16):
#                     B[z,y,0] = 10
#B = B.transpose((2,1,0)) 

#scopy = np.zeros(6,6,6)
#for k in range(6):
#       for j in range(6):
#              for i in range(6):
#                     scopy[j,i,k] = np.sqrt(i**5+k**2+j**2)

               #s[k,j,i] = np.sqrt(i**3+k**2+j**

#s = s.transpose()

#sys.exit(0)

#print(s)
#print(xx)
#print("")
#print("")
#print("")
#print("")
#print("")
#print(np.round(s))
"""
#sys.exit(0)
"""
from mayavi import mlab
#print("")
#print("")
#print("")
#print("")
#s = np.sqrt(xx**3+yy**2+zz**2)
#print(np.round(s))
pts = mlab.points3d(xx, yy, zz, s,mask_points = 1,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1   ,scale_factor=1/(15*np.max(s)))
#pts2 = mlab.points3d(xx, yy, zz, s2,mask_points = 1,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1 ,scale_factor=1/(np.max(s)))
mlab.colorbar(orientation='vertical')
mlab.xlabel('[Mpc/h]')
pts.glyph.glyph.clamping = False
#pts2.glyph.glyph.clamping = False
mlab.outline(pts)
mlab.show()
sys.exit(0)

#nx, ny = (3, 2)
x = np.linspace(0, 2,3 )
#y = np.linspace(0, 1, ny)

xv, yv,zv = np.meshgrid(x, x,x)
#xv, yv,zx = np.mgrid[0:90:10j,0:9:2j]
#print(xv)
#print(yv)


# indices using xyx indexing. Note thay numpy uses zyx indexing
a = np.array([[["000","100","200"],["010","110","210"],["020","120","220"]],[["001","101","201"],["011","111","211"],["021","121","221"]],[["002","102","202"],["012","112","212"],["022","122","222"]]])
print(a)
print("")
#print(a[0,1,2])
print(xv)
#N = 10
#for i in range(N):
#    for j in range(N):
#        for k in range(N):
#            s_ijk[k,j,i] = s_xyz[]
#            


"""
np.array([[0. , 0.5, 1. ],
       [0. , 0.5, 1. ]])

np.array([[0.,  0.,  0.],
       [1.,  1.,  1.]])
"""

#import matplotlib.pyplot as plt
#plt.plot(xv, yv, marker='o', color='k', linestyle='none')
#plt.show()