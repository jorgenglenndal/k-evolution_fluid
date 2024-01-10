import numpy as np
import sys
import h5py
from mayavi import mlab




class plot_class:
    def __init__(self, filename,indexing="xyz"): 
        #print("")
        #self.filename = "lol"
        loaded_myarray_2 = np.loadtxt("/uio/hume/student-u23/jorgeagl/src/master/master_project/new_chi_500.txt")
        data = loaded_myarray_2.reshape(501, 501, 501)
        condi = np.isnan(data) #self.data[0][self.data!=np.nan]
        data = data[condi == False]
        self.data = []
        self.data.append(data)
        print(self.data)
        #print(np.shape(self.data))
        #print(np.shape(self.data))

        print("18")
        
        self.filename = filename
        self.indexing = indexing
        #print(self.filename)
        #sys.exit(0)
        #self.indexing_help = print("Start indexing help: Standard numpy indexing is xyz. A numpy array, arr, of shape (nz, ny, nx) will have indices x, y and z such that arr[z,y,x] will return the value of the element located at index z in the z-direction, index y in the y-direction and index x in the x-direction. Mayavi uses ijk indexing, i.e., arr[i,j,k] returns the value of the element located at index i in the x-direction, index j in the y-direction and index k in the z-direction. If the data array, read from the input file, uses xyz indexing, the code will rearange the data correctly if and only if the parameter 'indexing' is set to its default value 'xyz'. Regarding gevolution: gevolution uses ijk indexing internally, but when the outputted HDF5 files are read in python, the data is in xyz indexing. End indexing help.")
        print("")
        #self.indexing_help = print(indexing_help)
        #self.data = []
        #self.shape = []
        #print(str(len(self.filename))+ " files.")
        #for i in range(len(self.filename)):
        #    self.data.append(self.load_data(i))
         
        self.shape = (501,501,501) #np.shape(self.data[0]) # all datasets should have the same shape
        self.n_part = 501 # assuming number of elements per dimension is in 0'th element
        
      
        
        #print("Number of elements in data = "+ str(len(self.data.flatten())))
        #print("n_part = " + str(self.n_part))
        #sys.exit(0)
        #self.x = np.linspace(0,1,self.n_part)
        #self.yy,self.xx,self.zz = np.meshgrid(self.x,self.x,self.x)
        #print(self.data)
        #self.xx,self.yy,self.zz = np.transpose(self.xx),np.transpose(self.yy),self.zz
        self.pc  = 3.08e16
        self.Mpc = self.pc*10**6

        self.h   = np.linspace(0.5, 1, 501)
        self.H0  = self.h*1e5/self.Mpc # kms^-1/Mpc
        H0_min = int(np.min(self.H0))
        H0_max = int(np.max(self.H0))
        len_array = complex(0, self.n_part)
        self.xx, self.yy, self.zz = np.mgrid[0:1:len_array, 0:1:len_array, 0:73:len_array]
        self.xx = self.xx[condi==False]
        self.yy = self.yy[condi==False]
        self.zz = self.zz[condi==False]
        self.ScalarData  = False
        self.VectorData = False
        self.mask = False
        self.split = False
        self.plot = False
        self.log_scale_bool = False
        #self.oneD = False

        if len(self.shape) == 3:
            self.ScalarData = True
            self.vmax = []
            for i in range(len(self.filename)):
                if self.indexing == "xyz":
                    self.data[i] = self.data[i].transpose((2, 1, 0)) # transposing to get ijk indexing
                self.vmax.append(np.max(abs(self.data[i].flatten())))
            print("")
            print("Working with scalar data")
            print("")           
        
        elif len(self.shape) == 4:
            self.VectorData = True
            print("")
            print("Working with vector data")
            print("Indexing probably wrong...Aborting...")
            sys.exit(1)
            self.data_x = self.data[:,:,:,0]
            self.data_y = self.data[:,:,:,1]
            self.data_z = self.data[:,:,:,2]
            print("")
        else:
            print("Data shape not accepted. Aborting...")
            #sys.exit(1)

        #if indexing == "xyz" and self.ScalarData:
        #    print("Transposing data...")
        #    for i in range(len(filename)):
        #        self.data[i] = self.data[i].transpose((2, 1, 0))
        print("94")
     
    

    def load_data(self,i,print_shape=True,print_h5_data_info=False):
        def myFunc(name, obj):
            if print_h5_data_info:
                print(name,obj)
        with h5py.File(self.filename[i], "r") as file:
            file.visititems(myFunc)
            data = file["data"][:]
            if print_shape:
                print("Shape of " + self.filename[i] +" = " +str(np.shape(data)))
        return data

    def mask_func(self,percentage=0.5,percentile=(0.25,99.75,"outside"),method = "rng",seed = 1234): # percentage of total number of points to plot
        self.masking_method = method
        ###default_random_name = "./" + self.filename_variable + "_random.npy"
        #if use_default_random_name: self.default_random_name = "./n_part_" + str(self.n_part) +  "_"+ self.masking_method + "_random.npy" 
        #self.percentage = percentage
        #self.n_part = n_part
        self.mask = True
        print("Masking...")
        if self.split == True:
            print("Warning: Masking must be done before splitting. Aborting...")
            sys.exit(1)
        #if load_random:
         #   print("Loading random")

        
        #if self.masking_method == "secrets":
        #    print("Using 'secrets' for random sampling. Don't...")
        #    if seed != False: print("Friendly Warning: Seed is never used in the 'secrets' method. You can instead save the matrix containg the random numbers")
        #    import secrets
        #    random = []
        #    for i in range(self.n_part**3):
        #       random.append(secrets.randbelow(self.n_part**3))
        #    random = np.array(random)
        #    random = random.reshape(self.shape)
        #    condition  = (random >= (self.n_part**3)*(100-percentage)/100)
        
        if self.masking_method == "rng":
            print("Keeping "+ str(percentage)+"% " "of the data")
            if seed == False:
                print("Using 'numpy.random.default_rng()' for uniform random sampling")
                rng = np.random.default_rng()
            else:
                print("Using 'numpy.random.default_rng(seed)' for uniform random sampling, where seed = " +str(seed))
                rng = np.random.default_rng(seed)
                
            random = rng.random(size=self.n_part**3)   
            random = random.reshape(self.shape)
            self.condition = (random >= (100-percentage)/100)
            actual_percentage = len((self.condition[self.condition]).flatten())/self.n_part**3*100
            print(f"Actual percentage of data kept: {actual_percentage:.3f}")
        
        elif self.masking_method== "limits":
            if self.VectorData: print("method: 'limits' does not work on vector data. Aborting..."); sys.exit(1)
            if seed != False: print("Friendly Warning: Seed is never used in the 'limits' method")

            percentile_bottom = float(percentile[0])
            percentile_top    = float(percentile[1])
            area = percentile[2]
            print("Keeping values '" + area + "' the range. Limits of range are exclusive.")
            percentile_bottom_value = np.percentile(self.data.flatten(),percentile_bottom)
            percentile_top_value = np.percentile(self.data.flatten(),percentile_top)
            if area == "outside":
                self.condition = (self.data < percentile_bottom_value) | (self.data > percentile_top_value) #np.where((self.data <= percentile_bottom_value) | (self.data >= percentile_top_value),True,False)
               
            elif area == "inside":
                self.condition = (self.data > percentile_bottom_value) & (self.data < percentile_top_value)
                
            else:
                print("Third element in 'percentile' must be 'outside' or 'inside'. Aborting...")
                sys.exit(1) 
            actual_percentage = len((self.condition[self.condition]).flatten())/self.n_part**3*100
            print(f"Percentage of data kept: {actual_percentage:.3f}")           

        else:
            print(str(self.masking_method) + " is not a method in mask_func. Aborting...")
            sys.exit(1)
        
        #if save_random:
        #    np.save(self.default_random_name if use_default_random_name else filename,self.condition)
        #    data_shape = self.shape if self.ScalarData else str(np.shape(self.data_x))
        #    print("Saving matrix with shape " + str(data_shape) +  ", containing True or False, to file: " + self.default_random_name)
        
        #check shape here!!! what if True goes to the wrong index
        #print(self.condition)
        #sys.exit(0)
        #print(self.xx[self.condition])

        self.xx = self.xx[self.condition]
        self.yy = self.yy[self.condition]
        self.zz = self.zz[self.condition]

        #self.xx = np.where(self.condition,self.xx,np.nan)
        #self.yy = np.where(self.condition,self.yy,np.nan)
        #self.zz = np.where(self.condition,self.zz,np.nan)
        
        if self.ScalarData:    
            for i in range(len(self.filename)):
                self.data[i] = self.data[i][self.condition]

            #self.data = np.where(self.condition,self.data,np.nan)
            #print(np.shape(self.data))
            #sys.exit(0)
        
        if self.VectorData:
            self.data_x       = self.data_x[self.condition]
            self.data_y       = self.data_y[self.condition]
            self.data_z       = self.data_z[self.condition]

        print("")
        
    def log_scale(self):
        print("Log scale")
        self.log_scale_bool = True
    
    def split_data(self):
        print("Splitting...")
        if self.mask == False:
            print("Friendly Warning: You have not masked. Continuing...")
        self.cond_upper = []
        self.cond_lower = []
        self.vmax_lower = []
        self.vmax_upper = []
        self.vmin_lower = []
        self.vmin_upper = []
        self.vmax       = []
        self.vmin       = []
        self.data_upper = []
        self.data_lower = []
        self.xx_upper   = []
        self.xx_lower   = []
        self.yy_upper   = []
        self.yy_lower   = []
        self.zz_upper   = []
        self.zz_lower   = []
        
        for i in range(len(self.filename)):
            if self.log_scale_bool:
                self.cond_upper.append(self.data[i] > 0)
            else:
                self.cond_upper.append(self.data[i] >= 0)  # & (delta_rho_fluid < 0.0010001)
            self.cond_lower.append(self.data[i] < 0)   #   & (-delta_rho_fluid <0.0010001)
            if len(self.cond_lower[i][self.cond_lower[i]]) == 0:
                print("No strictly negative data in file " +str(i+1)+  ". Aborting...")
                sys.exit(1)
            if len(self.cond_upper[i][self.cond_upper[i]]) == 0:
                print("No positive data in file " +str(i+1)+  ". Aborting...")
                sys.exit(1)
            self.vmax_lower.append(-np.min(self.data[i][self.cond_lower[i]])) # largest absolute value of the negative numbers
            self.vmax_upper.append(np.max(self.data[i][self.cond_upper[i]]))
            self.vmin_lower.append(-np.max(self.data[i][self.cond_lower[i]]))
            self.vmin_upper.append(np.min(self.data[i][self.cond_upper[i]])) 
            #print(self.vmax_lower)
            self.vmax.append(self.vmax_upper[i] if self.vmax_upper[i] >= self.vmax_lower[i] else self.vmax_lower[i])
            self.vmin.append(self.vmin_upper[i] if self.vmin_upper[i] <= self.vmin_lower[i] else self.vmin_lower[i])
            self.data_upper.append(self.data[i][self.cond_upper[i]])
            self.data_lower.append(-self.data[i][self.cond_lower[i]])
            self.xx_upper.append(self.xx[self.cond_upper[i]])
            self.xx_lower.append(self.xx[self.cond_lower[i]])
            self.yy_upper.append(self.yy[self.cond_upper[i]])
            self.yy_lower.append(self.yy[self.cond_lower[i]])
            self.zz_upper.append(self.zz[self.cond_upper[i]])
            self.zz_lower.append(self.zz[self.cond_lower[i]])

            #print("Number of elements in upper = "+str(len(self.cond_upper[self.cond_upper])))
            #print("Number of elements in lower = "+str(len(self.cond_lower[self.cond_lower])))
            self.split = True      
            print("")
    
    def plot_threeD_quiver(self,upscale_factor):
        #mlab.clf()
        obj = mlab.quiver3d(self.xx, self.yy, self.zz, self.data_x, self.data_y, self.data_z,scale_mode="vector",scale_factor = 1/self.n_part*upscale_factor,mode = "arrow")#,colormap = "inferno")
        #mlab.vectorbar(object=obj,orientation='vertical')
        mlab.colorbar(orientation='vertical')

        obj.glyph.glyph.clamping = False
        #obj.module_manager.vector_lut_manager.vector_bar.orientation = 'vertical'
        mlab.outline(obj)
        self.plot = True
        

    def plot_threeD_scatter(self,i = 0,upscale_factor=1):
        print("Plotting...")
        #fig = mlab.figure()
        scale_factor = upscale_factor/(self.vmax[i]*(self.n_part - 1))#1/self.n_part*upscale_factor
        if self.mask == False:
            print("Friendly Warning: You have not masked the data. Continuing...")
        if self.split:
            #scale_factor_upper = scale_factor #if self.vmax_upper >= self.vmax_lower else scale_factor*self.vmax_upper/self.vmax
            #scale_factor_lower = scale_factor #if self.vmax_lower > self.vmax_upper else scale_factor*self.vmax_lower/self.vmax
            self.obj_upper = mlab.points3d(self.xx_upper[i],self.yy_upper[i],self.zz_upper[i],self.data_upper[i],mask_points = 1,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = 'Reds',scale_factor=scale_factor,vmin = self.vmin[i] if self.log_scale_bool else 0, vmax=self.vmax[i] )#,extent=[0, 1, 0, 2, 0, 3])
            self.obj_lower = mlab.points3d(self.xx_lower[i],self.yy_lower[i],self.zz_lower[i],self.data_lower[i],mask_points = 1,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = 'Blues',scale_factor=scale_factor,vmin = self.vmin[i] if self.log_scale_bool else 0,vmax=self.vmax[i])#,extent=[0, 1, 0, 2, 0, 3])
            mlab.colorbar(object=self.obj_upper,orientation='vertical')#,title= "overdensity for velocity")
            mlab.colorbar(object=self.obj_lower,orientation='vertical')#,title= "overdensity for velocity")
            self.obj_upper.module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([0.01,  0.1])
            self.obj_lower.module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([0.89,  0.1])
            #obj_lower.module_manager.scalar_lut_manager.scalar_bar_actor.position = "precede" 

            self.obj_upper.glyph.glyph.clamping = False
            self.obj_lower.glyph.glyph.clamping = False
            mlab.outline(self.obj_upper)
            #mlab.xlabel('1')# [Mpc/h]')
            #mlab.ylabel('2')# [Mpc/h]')
            #mlab.zlabel('3')# [Mpc/h]')
            #ax = mlab.gcf()
            #ax.view_init()
            #mlab.view(distance='auto', focalpoint=[0, 0, 0])
            #mlab.view(azimuth=270)
            #mlab.view(azimuth=0, elevation=90)
            #v = mlab.view()
            #print(v)
            #mlab.show()
        else:
            self.obj = mlab.points3d(self.xx,self.yy,self.zz,self.data[i],mask_points = 100,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,scale_factor=scale_factor)
            mlab.colorbar(object=self.obj,orientation='vertical')#,title= "overdensity for velocity")
            self.obj.glyph.glyph.clamping = False
            mlab.outline(self.obj)
            #mlab.xlabel('300 [Mpc/h]')
            #mlab.ylabel('300 [Mpc/h]')
            #mlab.zlabel('300 [Mpc/h]')
            #mlab.show()
        self.plot = True
    def show(self):
        if self.plot: mlab.show()
        print("Nothing to show")
    
    def help(self,Q):
        if Q == "indexing":
            print("")
            print("Help: Standard numpy indexing is xyz. A numpy array, arr, of shape (nz, ny, nx) will have indices x, y and z such that arr[z,y,x] will return the value of the element located at index z in the z-direction, index y in the y-direction and index x in the x-direction. Mayavi uses ijk indexing, i.e., arr[i,j,k] returns the value of the element located at index i in the x-direction, index j in the y-direction and index k in the z-direction. If the data array, read from the input file, uses xyz indexing, the code will rearange the data correctly if and only if the parameter 'indexing' is set to its default value 'xyz'. Regarding gevolution: gevolution uses ijk indexing internally, but when the outputted HDF5 files are read in python, the data is in xyz indexing. End help.")
            print("")
        elif Q == "method":
            print("")
            print("Help: Possible options for 'method' is 'rng' and 'limits'. 'rng' needs the parameter 'percentage' to determine how many percent of the data to keep. 'limits' needs a parameter 'percentile' on the form [a,b,c], where a is the bottom percentile, of interest, of the data and b is the top percentile, of interest, of the data. c must be either 'outside' or 'inside' the limits. Limits are excluded from the resulting dataset. End help. ")
            print("")
        else:
            print("...")
    def move_camera(self,obj):
        def elevation(direction = "positive"):
            if direction == "positive":
                return 0.25
            else:
                return -0.25


        #@mlab.show
        @mlab.animate(delay=10)
        def anim():
            #f = mlab.gcf()
            mlab.view(113.24999999999946, 15.485610317245463, 4, np.array([0.50011478, 0.50011478, 0.50011774]))

            elevation_turn_bool = True
            while 1:
                obj.scene.camera.azimuth(0.25)
                obj.scene.camera.elevation(elevation(direction = "positive" if elevation_turn_bool == False else "negative"))
                
                obj.scene.camera.zoom(1.005 if elevation_turn_bool==True else 0.995)
                obj.scene.render()
                if mlab.view()[1] <=1: elevation_turn_bool = True
                if mlab.view()[1] >=179: elevation_turn_bool = False

                #print(mlab.view()[:])
                yield
        anim()




#s = np.zeros((16,16,16)) # zyx
##B = np.zeros((16,16,16))
#for k in range(16):
#       for j in range(16):
#              for i in range(16):
#                     #s[i,j,0] = 10
#                     s[i,j,k] = 100000*np.sqrt(i**2+k**2+j**2)  #str(i) +"_" + str(j) + "_"+str(k)

#test = plot_class(["1.h5","2.h5","3.h5"])

#test = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid1.h5")
#test.plot_threeD_scatter(1)
#test.animate_camera()
#test.show()

#loaded_myarray_2 = np.loadtxt("/uio/hume/student-u23/jorgeagl/src/master/master_project/new_chi_500.txt")
#new_chi_load = loaded_myarray_2.reshape(501, 501, 501)

test = plot_class(["lol"],"ijk")

#test = plot_class(["/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/snap_001_delta_rho_fluid.h5"])
#test1 = plot_class(["/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid1.h5","/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid1.h5"])
#test1.mask_func(percentage=50)
#test.mask_func(percentage=1)
#test.split_data() # should always split to avod color bug
test.plot_threeD_scatter(upscale_factor=10)
#test.move_camera(test.obj_upper)
test.show()
sys.exit(0)
#test2 = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid_5.h5")
#test3 = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid_10.h5")
#test4 = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid_15.h5")
#test5 = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid_20.h5")
#test6 = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid_25.h5")
#test7 = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid_30.h5")
#test8 = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid_35.h5")
#test9 = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid_40.h5")


#test = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid2.h5")
#test = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid1.h5")
#test.indexing_help
#sys.exit(0)
#print(test.data[8,3,2]) # z,y,x
#sys.exit(0)
#test.mask_func(method="limits",percentile=[1,99,"outside"],percentage=1,seed = 12345)
#test.mask_func(percentage=5)
#test = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/snap_001_v_upper_i_fluid.h5")
#test.split_data()
#test.plot_threeD_scatter(upscale_factor=1)#800)
#print(test.data)
#test.show()
#test.help("indexing")
#test.help("method")
#test.log_scale()
#test.plot_threeD_scatter(upscale_factor=1)#800)
#data1 = test1.data
#data2 = test2.data
#data3 = test3.data
#data4 = test4.data
#data5 = test5.data
#data6 = test6.data
#data7 = test7.data
#data8 = test8.data
#data9 = test9.data
#datas = np.array([data1,data2,data3,data4,data5,data6,data7,data8,data9])

#x,y,z = test1.xx,test1.yy,test1.zz
#
#test1.plot_threeD_scatter(1)
#test1.animate()
#test1.show()
sys.exit(0)

#plot = mlab.points3d(x,y,z,data1)
#ms = l.mlab_source
"""
@mlab.animate
def anim():
    for i in range(9):
        plot.mlab_source.scalars = datas[i]
        yield
anim()
mlab.show()
"""
@mlab.show
@mlab.animate
def anim():
    #f = mlab.gcf()
    while 1:
        f.scene.camera.azimuth(10)
        f.scene.render()
        yield

anim()
"""
from tvtk.tools import visual
from vtk.util import colors as color

# Create a figure
f = mlab.figure()#size=(200,200))
# Tell visual to use this as the viewer.
visual.set_viewer(f)

@mlab.show
@mlab.animate(delay=500)
def anim():
    while 1:
        for i in range(9):
            #b1.x = b1.x + b1.v*0.01
            #if b1.x > 3 or b1.x < -3:
            #    b1.v = -b1.v
            plot.mlab_source.scalars = datas[i]
            yield


anim()
"""
#test.help("indexing")
#sys.exit(0)
#test = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/snap_001_v_upper_i_fluid.h5")
#test = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid.h5")
#test1 = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/delta_rho_fluid1.h5")

#test.plot_threeD_scatter(9)
#test.show()
#cond = (test.data == test1.data)
#if (test1.data == 1.0).any(): print("true")
#print(test1.data)
sys.exit(0)


#gevolutino used xyz. Numpy uses zyx. Converting to ijk
data_xyz = np.zeros((test1.n_part,test1.n_part,test1.n_part))
data_ijk = np.zeros((test1.n_part,test1.n_part,test1.n_part))
for z in range(test1.n_part):
    for y in range(test1.n_part):
        for x in range(test1.n_part):
            # converting from zyx to xyz
            data_xyz[x,y,z] = test1.data[z,y,x]

for k in range(test1.n_part):
    for j in range(test1.n_part):
        for i in range(test1.n_part):
            
            data_ijk[i,j,k] = test1.data[j,i,k]
            # converting from xyz to ijk
            #copy_ijk[i,j,k] = copy_xyz[j,i,k]
#np.savetxt("./rho_test.h5",data_ijk)
print(data_ijk)



#print(test1.data[0])

#print(test.data_z)
#test.mask_func(percentage=0.5,percentile = (1,99,"outside"),method = "rng",seed = False)
#for i in range(len(test.data.flatten())):
#    print(test.data.flatten()[i])
#print(test.data)
#np.savetxt("./lol.txt",test.data.flatten())
#print(test.data)

#test.maskfunc_loaded_random(percentage=0.5,method="rng",filename = "example.npy",use_default_random_name = True)
#test.split_data()
#test.plot_threeD_scatter(upscale_factor=9)
##test.plot_threeD_quiver(upscale_factor=9)
#test.show()





#test = plot_class("lol.h5")


#test_inst = plot_class("/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/snap_001_delta_rho_fluid.h5")
#test_inst.mask_func(10)
##test_inst.mask_func_loaded_random(1,"./random.npy")
#test_inst.split_data()
#test_inst.plot_threeD_scatter(9)
#test_inst.show()



#print(np.shape(v_fluid))
#ex = plot_class(delta_rho_fluid)
##print(len(ex.data.flatten()))
##ex.mask_func(1)
##print(len(ex.data.flatten()))
#ex.mask_func_loaded_random(1,"./")
##ex.threeD_quiver()
#ex.split_data()
#ex.threeD_scatter(9)
#ex.show()
#sys.exit(0)



