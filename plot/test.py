import sys
import numpy as np

#a = np.linspace(0,1,11)
#cond = a > 0.5
#b = a[cond]
#b[0] = 100
##print(b[0] == a[0]) 
#print(a)      
##[[0,0,0],[0,0,0],[0,0,0]]
#test = np.array([[[0,0,0],[0,0,0],[0,0,0]],[[1,1,1],[1,1,1],[1,1,1]],[[2,2,2],[2,2,2],[2,2,2]]])
#np.save("./test_indexing",test)
test = np.load("./test_indexing.npy")
print(test)
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