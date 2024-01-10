import numpy as np
import sys
import h5py
from mayavi import mlab




class plot_class:
    def __init__(self,filename,indices="all",data_indexing="xyz",verbose=True):
        self.verbose_bool = verbose
        if self.verbose_bool: print("")
        if isinstance(filename, list) == False:
            if self.verbose_bool: print("1 file in 'filename' is assumed. If incorrect, you must convert 'filename' to a list.")
            self.filename = [filename]
        else:
            self.filename = filename
            if self.verbose_bool: print(str(len(self.filename))+ " file(s) in 'filename'.")
        if len(self.filename) > 1:
            if self.verbose_bool: print("The shape of the data is assumed to be the same in all files. The shape is read from the first file.")

        self.indexing = data_indexing
        
        if isinstance(indices, list)==False:
            if indices == "all":
                self.indices = [indices]
            else:
                print("'indices' must be a list. Aborting...")
                sys.exit(1)
        else:
            self.indices = indices
        if self.verbose_bool: print("")
        
        self.data = self.load_hdf5_data(0)
        self.shape = np.shape(self.data) # all datasets should have the same shape

        if self.shape[0] == self.shape[1] == self.shape[2]:
            self.n_grid = self.shape[0] # assuming number of elements per dimension is in 0'th element
        else:
            print("Number of grid points is not the same in each dimension. For vector data, shape must be (x,x,x,y), where y are the vectors belonging to the grid points. Aborting...")
            sys.exit(1)
        
        len_array = complex(0, self.n_grid)
        self.xx, self.yy, self.zz = np.mgrid[0:1:len_array, 0:1:len_array, 0:1:len_array]
        self.ScalarData  = False
        self.VectorData = False
        self.mask_bool = False
        self.plot_bool = False
        self.log_scale_bool = False
        self.symmetric_colorbar_bool = False
        self.scatter_bool = False
        self.move_camera_bool = False
        self.save_bool = False
        self.offscreen_rendering_bool = False
        self.show_bool = False
        self.rng_generated_bool = False
        #self.scatter_mode = "standard"
        #self.verbose_bool = False

        if len(self.shape) == 3:
            self.ScalarData = True 
            if self.verbose_bool: print("")
            if self.verbose_bool: print("Working with scalar data.")
            if self.verbose_bool: print("")           
        
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
            sys.exit(1)
        
    def load_hdf5_data(self,i,print_shape = True,print_h5_data_info=False):
        with h5py.File(self.filename[i],'r') as file:
            keys = list(file.keys())
            if print_h5_data_info:
                print(keys)
                print(type(file[keys[0]]))
            if len(keys) != 1:
                print("Wrong format in HDF5 file. File must contain one dataset. Modyfy the 'load_hdf5_data' function. Aborting...")
                sys.exit(1)

            data = file[keys[0]][()]  # returns as a numpy array. See hdf5 documentation online

            #print(type(file[keys[0]])) 
        if self.verbose_bool and print_shape:
            print("Shape of " + self.filename[i] +" = " +str(np.shape(data)))
        return data
    
    def mask(self,method = "rng",percentage=0.5,percentile=(0.25,99.75,"outside"),seed = 1234):
        self.mask_bool = True
        self.mask_method = method
        if method =="rng":
            self.mask_seed = seed
            self.mask_percentage = percentage
        elif method == "limits":
            self.mask_percentile = percentile
        else:
            print("Unknown 'method'. Aborting...")
            sys.exit(1)
    
    def symmetric_colorbar(self):
        self.symmetric_colorbar_bool = True
        if self.verbose_bool: print("Symmetric colorbar selected.")
    def log_scale(self,method = "split"):
        self.log_scale_method = method
        if self.verbose_bool: print("Log scale selected with method '" + self.log_scale_method + "'.")
        self.log_scale_bool = True
    def scatter(self,rescale_factor=1):
        self.scatter_bool = True
        self.rescale_factor = rescale_factor
        if self.verbose_bool: print("Scatter plot selected.")
    def save(self):
        self.save_bool = True
        if self.verbose_bool: print("File(s) will be saved.")
    def offscreen_rendering(self):
        self.offscreen_rendering_bool = True
        if self.verbose_bool: print("")
        if self.verbose_bool: print("Off-screen rendering enabled. May be much slower than on-screen rendering.")
        if self.verbose_bool: print("")
    def show(self):
        self.show_bool = True
        #print("Will show.")
    def move_camera(self):
        self.move_camera_bool = True


    def mask_func(self):
        if self.verbose_bool: print("Masking...")
        
        if self.mask_method == "rng":
            if self.verbose_bool: print("Keeping "+ str(self.mask_percentage)+"% " "of the data")
            if self.mask_seed == False:
                if self.verbose_bool: print("Using 'numpy.random.default_rng()' for uniform random sampling")
                rng = np.random.default_rng()
            else:
                if self.verbose_bool: print("Using 'numpy.random.default_rng(seed)' for uniform random sampling, where seed = " +str(self.mask_seed))
                rng = np.random.default_rng(self.mask_seed)
                
            random = rng.random(size=self.n_grid**3)   
            random = random.reshape(self.shape)
            self.condition = random >= (100-self.mask_percentage)/100
            actual_percentage = len((self.condition[self.condition]).flatten())/self.n_grid**3*100
            if self.verbose_bool: print(f"Actual percentage of data kept: {actual_percentage:.3f}")
            self.rng_generated_bool = True
        
        elif self.mask_method== "limits":
            if self.VectorData: print("method: 'limits' does not work on vector data. Aborting..."); sys.exit(1)
            #if self.mask_seed != False: print("Friendly Warning: Seed is never used in the 'limits' method")

            percentile_bottom = float(self.mask_percentile[0])
            percentile_top    = float(self.mask_percentile[1])
            area              = str(self.mask_percentile[2])
            if self.verbose_bool: print("Keeping values '" + area + "' the range. The limits of the range are exclusive for 'outside' and inclusive for 'inside'.")
            percentile_bottom_value = np.percentile(self.data.flatten(),percentile_bottom)
            percentile_top_value = np.percentile(self.data.flatten(),percentile_top)
            if area == "outside":
                self.condition = (self.data < percentile_bottom_value) | (self.data > percentile_top_value) #np.where((self.data <= percentile_bottom_value) | (self.data >= percentile_top_value),True,False)
               
            elif area == "inside":
                self.condition = (self.data >= percentile_bottom_value) & (self.data <= percentile_top_value)
                
            else:
                print("Third element in 'percentile' must be 'outside' or 'inside'. Aborting...")
                sys.exit(1) 
            actual_percentage = len((self.condition[self.condition]).flatten())/self.n_grid**3*100
            if self.verbose_bool: print(f"Percentage of data kept: {actual_percentage:.3f}")           

        else:
            print(str(self.mask_method) + " is not a method in mask_func. Aborting...")
            sys.exit(1)
        
        """
        self.xx_current = self.xx[self.condition]
        self.yy_current = self.yy[self.condition]
        self.zz_current = self.zz[self.condition]

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
        """
        print("")
    
    #if self.verbose_bool: print("Symmetric colorbar has been auto-deselected.")
    def symmetric_colorbar_func(self):
        if self.log_scale_bool:
            if self.log_scale_method == "abs":
                if self.verbose_bool: print("Symmetric colorbar has been auto-deselected.")
                return
            if self.scatter_mode == "only positive":
                if self.verbose_bool: print("Symmetric colorbar has been auto-deselected for this file.")
                return
            if self.scatter_mode == "only negative":
                if self.verbose_bool: print("Symmetric colorbar has been auto-deselected for this file.")
                return
            if self.scatter_mode == "split":
                ###self.max_positive = np.nanmax(self.positive_data.flatten())
                ###self.max_negative = np.nanmax(self.negative_data.flatten())
                ###self.min_positive = np.nanmin(self.positive_data.flatten())
                ###self.min_negative = np.nanmin(self.negative_data.flatten())
                
                if self.max_positive >= self.max_negative:
                    self.max_log = self.max_positive
                    #self.max_log_positive = max_positive
                else:
                    self.max_log = self.max_negative
                    #self.max_log_upper = max_positive
                
                if self.min_positive <= self.min_negative:
                    self.min_log = self.min_positive
                else:
                    self.min_log = self.min_negative
                
        else:
            min = np.nanmin(self.data.flatten())
            max = np.nanmax(self.data.flatten())
            if abs(max) >= abs(min):
                self.symmetric_limit = abs(max)
            else:
                self.symmetric_limit = abs(min)

            self.vmin = -self.symmetric_limit
            self.vmax = self.symmetric_limit
            self.abs_max = self.symmetric_limit
         
    def plot_threeD_quiver(self,upscale_factor):
        #mlab.clf()
        obj = mlab.quiver3d(self.xx, self.yy, self.zz, self.data_x, self.data_y, self.data_z,scale_mode="vector",scale_factor = 1/self.n_grid*upscale_factor,mode = "arrow")#,colormap = "inferno")
        #mlab.vectorbar(object=obj,orientation='vertical')
        mlab.colorbar(orientation='vertical')

        obj.glyph.glyph.clamping = False
        #obj.module_manager.vector_lut_manager.vector_bar.orientation = 'vertical'
        mlab.outline(obj)
        self.plot = True
        
    def scatter_func(self):
        if self.log_scale_bool == False: scale_factor = self.rescale_factor/(self.abs_max*(self.n_grid - 1))
        else:
            scale_factor = self.rescale_factor/(self.n_grid - 1)

        if self.symmetric_colorbar_bool:
            colormap = 'seismic'
            
            "I was not able to implement 'colorcet' in 'mayavi'."
            #import colorcet as cc
            #from matplotlib.cm import get_cmap

            #colormap = str(get_cmap("cet_fire"))
            #colormap = cc.m_rainbow4

        else:
            colormap = 'jet'

        if self.log_scale_bool:
            if self.verbose_bool: print("Clamping the glyphs. From the mayavi documentation: 'the smallest value of the scalar data is represented as a null diameter, and the largest is proportional to inter-point distance.' The documentation: https://docs.enthought.com/mayavi/mayavi/mlab.html")
            if self.log_scale_method == "abs":
                colormap = 'jet'
                self.obj = mlab.points3d(self.xx_current,self.yy_current,self.zz_current,self.data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor)#,vmin = self.vmin, vmax = self.vmax)#,extent=[0, 1, 0, 2, 0, 3])
                self.obj.glyph.glyph.clamping = True
                self.obj.actor.property.representation = "wireframe" # colorbug occurs for option: "surface"
                mlab.colorbar(object=self.obj,orientation='vertical')#,title= "overdensity for velocity")
                mlab.outline(self.obj)
            elif self.log_scale_method == "split":
                if self.scatter_mode == "only positive":
                    colormap = "Reds"
                    self.obj = mlab.points3d(self.xx_positive,self.yy_positive,self.zz_positive,self.positive_data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor)#,vmin = self.vmin, vmax = self.vmax)#,extent=[0, 1, 0, 2, 0, 3])
                    self.obj.glyph.glyph.clamping = True
                    self.obj.actor.property.representation = "wireframe" # colorbug occurs for option: "surface"
                    mlab.colorbar(object=self.obj,orientation='vertical')#,title= "overdensity for velocity")
                    mlab.outline(self.obj)

                elif self.scatter_mode == "only negative":
                    colormap = "Blues"
                    self.obj = mlab.points3d(self.xx_negative,self.yy_negative,self.zz_negative,self.negative_data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor)#,vmin = self.vmin, vmax = self.vmax)#,extent=[0, 1, 0, 2, 0, 3])
                    self.obj.glyph.glyph.clamping = True
                    self.obj.actor.property.representation = "wireframe" # colorbug occurs for option: "surface"
                    mlab.colorbar(object=self.obj,orientation='vertical')#,title= "overdensity for velocity")
                    mlab.outline(self.obj)
                elif self.scatter_mode == "split":
                    #if self.positive_data.flatten().all() >=1:
                    colormap = "Reds"
                    self.obj_positive = mlab.points3d(self.xx_positive,self.yy_positive,self.zz_positive,self.positive_data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor if self.max_positive >= self.max_negative else scale_factor*self.max_positive/self.max_negative,vmin = self.min_log if self.symmetric_colorbar_bool else None, vmax = self.max_log if self.symmetric_colorbar_bool else None)#,extent=[0, 1, 0, 2, 0, 3])
                    self.obj_positive.glyph.glyph.clamping = True
                    self.obj_positive.actor.property.representation = "wireframe" # colorbug occurs for option: "surface"
                    mlab.colorbar(object=self.obj_positive,orientation='vertical')#,title= "overdensity for velocity")
                    mlab.outline(self.obj_positive)

                    colormap = "Blues"
                    self.obj_negative = mlab.points3d(self.xx_negative,self.yy_negative,self.zz_negative,self.negative_data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor if self.max_negative >= self.max_positive else scale_factor*self.max_negative/self.max_positive,vmin = self.min_log if self.symmetric_colorbar_bool else None, vmax = self.max_log if self.symmetric_colorbar_bool else None)#,vmin = self.vmin, vmax = self.vmax)#,extent=[0, 1, 0, 2, 0, 3])
                    self.obj_negative.glyph.glyph.clamping = True
                    self.obj_negative.actor.property.representation = "wireframe" # colorbug occurs for option: "surface"
                    mlab.colorbar(object=self.obj_negative,orientation='vertical')#,title= "overdensity for velocity")
                    #mlab.outline(self.obj_negative)

                    self.obj_positive.module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([0.01,  0.1])
                    self.obj_negative.module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([0.89,  0.1])

                else:
                    print("Error in scatter mode. Aborting...")
                    sys.exit(1)

            else:
                print("Unknown log scale method. Aborting...")
                sys.exit(1)

        else:
            self.obj = mlab.points3d(self.xx_current,self.yy_current,self.zz_current,self.data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor,vmin = self.vmin if self.symmetric_colorbar_bool else None, vmax = self.vmax if self.symmetric_colorbar_bool else None)
            self.obj.glyph.glyph.clamping = False 
            self.obj.actor.property.representation = "wireframe" # colorbug occurs for option: "surface"
            mlab.colorbar(object=self.obj,orientation='vertical')#,title= "overdensity for velocity")
            mlab.outline(self.obj)
        self.plot_bool = True
        
        

    """
    #def scatter_func_test(self,i = 0,upscale_factor=1):
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
            self.obj = mlab.points3d(self.xx,self.yy,self.zz,self.data[i],mask_points = 1,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,scale_factor=scale_factor)
            mlab.colorbar(object=self.obj,orientation='vertical')#,title= "overdensity for velocity")
            self.obj.glyph.glyph.clamping = False
            mlab.outline(self.obj)
            #mlab.xlabel('300 [Mpc/h]')
            #mlab.ylabel('300 [Mpc/h]')
            #mlab.zlabel('300 [Mpc/h]')
            #mlab.show()
        self.plot_bool = True
    """
    def plotting_loop_func(self):
        def inside_loop_func(i):
            mlab.figure(size=(800, 800))
            self.data = self.load_hdf5_data(i,print_shape=False) # reading all the data
            
            if self.indexing == "xyz":
                self.data = self.data.transpose((2, 1, 0)) # transposing to get ijk indexing   
            
            if self.mask_bool:
                if self.mask_method == "limits":
                    self.mask_func()
                    self.data = self.data[self.condition]
                    self.xx_current, self.yy_current, self.zz_current = self.xx[self.condition], self.yy[self.condition], self.zz[self.condition]
                elif self.mask_method == "rng":
                    if self.rng_generated_bool == False:
                        self.mask_func()
                        self.data = self.data[self.condition]
                        self.xx_current, self.yy_current, self.zz_current = self.xx[self.condition], self.yy[self.condition], self.zz[self.condition]
                    else:
                        self.data = self.data[self.condition]
                else:
                    print("Unknown mask method. Aborting...")
                    sys.exit(1)
                
            # data and coords have been masked
            if self.log_scale_bool:
                if self.log_scale_method == "abs":
                    self.data = np.log10(np.abs(self.data))
                    if self.symmetric_colorbar_bool:
                        if self.verbose_bool: print("Warning: you are trying to use a symmetric colorbar on an absolute value plot. Symmetric colorbar is disabled...")
                elif self.log_scale_method == "split":        
                    positive_booleans = self.data > 0
                    negative_booleans = self.data < 0
                
                    self.positive_data = self.data[positive_booleans]
                    self.negative_data = - self.data[negative_booleans]
                    
                    # avoiding empty matrices
                    if len(self.positive_data) + len(self.negative_data) == 0:
                        print("File with index, " + str(i) + ", is either empty or has no non-zero values. Skipping this file...")
                        return
                    elif len(self.positive_data) == 0:
                        self.scatter_mode = "only negative"
                        print("no positive")
                        self.xx_negative = self.xx_current[negative_booleans]
                        self.yy_negative = self.yy_current[negative_booleans]
                        self.zz_negative = self.zz_current[negative_booleans]
                        self.negative_data = np.log10(self.negative_data)
                    elif len(self.negative_data) == 0:
                        self.scatter_mode = "only positive"
                        print("no negative")
                        self.xx_positive = self.xx_current[positive_booleans]
                        self.yy_positive = self.yy_current[positive_booleans]
                        self.zz_positive = self.zz_current[positive_booleans]
                        self.positive_data = np.log10(self.positive_data)
                    else:
                        self.scatter_mode = "split"

                        self.xx_positive = self.xx_current[positive_booleans]
                        self.yy_positive = self.yy_current[positive_booleans]
                        self.zz_positive = self.zz_current[positive_booleans]
                        self.xx_negative = self.xx_current[negative_booleans]
                        self.yy_negative = self.yy_current[negative_booleans]
                        self.zz_negative = self.zz_current[negative_booleans]

                        self.positive_data = np.log10(self.positive_data)
                        self.negative_data = np.log10(self.negative_data)
                        self.max_positive = np.nanmax(self.positive_data.flatten())
                        self.max_negative = np.nanmax(self.negative_data.flatten())
                        self.min_positive = np.nanmin(self.positive_data.flatten())
                        self.min_negative = np.nanmin(self.negative_data.flatten())
                        if self.symmetric_colorbar_bool == False:
                            if self.verbose_bool: print("")
                            if self.verbose_bool: print("Symmetric colorbar is recommended to get more consistent colors. Scaling will be consistent.")
                            if self.verbose_bool: print("")
                    
                else:
                    print("Unknown log_scale method. Available methods are: 'abs', 'split'. Aborting...")
                    sys.exit(1)

            if self.symmetric_colorbar_bool:
                self.symmetric_colorbar_func()
            else:
                if self.log_scale_bool == False:
                    self.vmin = np.nanmin(self.data.flatten())
                    self.vmax = np.nanmax(self.data.flatten())
                    self.abs_max = abs(self.vmax) if abs(self.vmax) >= abs(self.vmin) else abs(self.vmin)
            
            if self.scatter_bool:
                self.scatter_func()
                #mlab.view(azimuth=None, elevation=None, distance=None, focalpoint=None,roll=None, reset_roll=True, figure=None)
                #mlab.view(146.2499999999997, 48.48561031724544, 3.9999999999999947, np.array([0.50011478, 0.50011478, 0.50011774]))
                if self.move_camera_bool == False: mlab.view(-119.00051786825493, 69.04300699344203, 2.5, np.array([0.50011478, 0.50011478, 0.50011774]))

            if self.plot_bool:
                #print(i)
                if self.mask_bool == False and self.verbose_bool: print("Plotting without masking... May be very slow.")
                if self.save_bool:
                    if int(len(self.filename)) > 10001:
                        print("Too many files. The mlab.savefig() line must be changed from 04d to e.g. 05d. Aborting...")
                        sys.exit(1)
                    if self.verbose_bool: print("Saving files in ./tmp_.png")
                    mlab.savefig("tmp_%04d.png" % i)
                if self.move_camera_bool:
                    self.move_camera_func()
                if self.show_bool: 
                    mlab.show()
                else:
                    mlab.close()
            else:
                if self.verbose_bool: print("Nothing has been plotted.")
            
        import copy 
        # defining coordinate matrices independent of self.xx, ...    
        self.xx_current, self.yy_current, self.zz_current =  copy.deepcopy(self.xx),copy.deepcopy(self.yy),copy.deepcopy(self.zz)   
        
        #new_object = copy.deepcopy(self.data)

        #self.xx_current[0,0,0] = 100
        #print(self.xx_current[0,0,0] == self.xx[0,0,0])  
        #sys.exit(1)
        #if self.mask_bool:
        #    if self.mask_method== "rng":
        #        self.mask_func() 
        #self.xx_current, self.yy_current, self.zz_current = self.xx, self.yy, self.zz
        
        if self.offscreen_rendering_bool:
            mlab.options.offscreen = True
        else:
            mlab.options.offscreen = False
        
        if self.indices[0] == "all" or self.indices[0] == "range":
            plot_last = False
            for i in range(self.start,self.stop,self.step):
                inside_loop_func(i)
                if i == self.stop-1:
                    plot_last = True
            if plot_last == False and self.indices[0] == "range":
                if self.verbose_bool: print("Plotting the last file in range --> index "+ str(self.stop-1))
                inside_loop_func(self.stop-1)

            
        elif self.indices[0] == "singles":
            for i in self.indices[1:]:
                inside_loop_func(i)
        else:
            import inspect

            line_number = inspect.currentframe().f_back.f_lineno
            print("'indices' invalid. Aborting...")
            print("Line number:", line_number)
            sys.exit(1)

    
    def execute(self):
        # checking booleans and doing tasks in correct order inside plotting_loop_func()...
        if self.verbose_bool: print("")
        if self.indices[0] == "all":
            if self.verbose_bool: print("'indices' set to 'all'.")
            self.start = 0
            self.stop  = len(self.filename)
            self.step  = 1
            self.plotting_loop_func()

        
        elif self.indices[0] == "singles":
            if self.verbose_bool: print("'indices' set to 'singles'.")
            self.plotting_loop_func()


        elif self.indices[0] == "range":
            if self.verbose_bool: print("'indices' set to 'range'.")
            self.start = int(self.indices[1])
            self.stop = int(self.indices[2])
            self.step = int(self.indices[3]) if len(self.indices) == 4 else 1
            self.plotting_loop_func()

        else:
            print("'indices' set incorrectly. Aborting...")
            sys.exit(1)    
        print("")                    
    
    def help_indexing(self):
        print("")
        print("Help: Standard numpy indexing is called 'xyz'. A numpy array, arr, of shape (n_z, n_y, n_x) will have indices x, y and z such that arr[z,y,x] will return the value of the element located at index z in the z-direction, index y in the y-direction and index x in the x-direction. Mayavi uses 'ijk' indexing, i.e., arr[i,j,k] returns the value of the element located at index i in the x-direction, index j in the y-direction and index k in the z-direction. If the data array, read from the input file, uses 'xyz' indexing, the code will rearange the data correctly if, and only if, the parameter 'indexing' is set to its default value: 'xyz'. Regarding gevolution: gevolution uses 'ijk' indexing internally, but when the outputted HDF5 files are read in python, the data is in 'xyz' indexing. End help.")
        print("")
        
    def help_method(self):
        print("")
        print("Help: Possible options for 'method' is 'rng' and 'limits'. 'rng' needs the parameter 'percentage' to determine how many percent of the data to keep. 'limits' needs a parameter 'percentile' on the form [a,b,c], where a is the bottom percentile, of interest, of the data and b is the top percentile, of interest, of the data. c must be either 'outside' or 'inside' the limits. Limits are excluded from the resulting dataset. End help. ")
        print("")

    #def move_camera(self):
    #    self.move_camera_bool = True
    def move_camera_func(self):
        def elevation(direction = "positive"):
            if direction == "positive":
                return 0.25
            else:
                return -0.25


        #@mlab.show
        @mlab.animate(delay=200,ui = False if self.offscreen_rendering_bool else True)
        def anim():
            #f = mlab.gcf()
            ###mlab.view(113.24999999999946, 15.485610317245463, 4, np.array([0.50011478, 0.50011478, 0.50011774]))
            #mlab.view(-85.5924006618464, 60.53116127610586, 4.000000000000003, np.array([0.50011478, 0.50011478, 0.50011774]))
            #mlab.view(-119.00051786825493, 69.04300699344203, 3, np.array([0.50011478, 0.50011478, 0.50011774]))
            mlab.view(-120.52364487631695, 66.68905768142035, 3, np.array([0.50011478, 0.50011478, 0.50011774]))


            #mlab.view(azimuth = 90,elevation=90)

            elevation_turn_bool = True
            while True:
                #obj.scene.camera.azimuth(0.25)
                #obj.scene.camera.elevation(elevation(direction = "positive" if elevation_turn_bool == False else "negative"))
                if self.log_scale_bool:
                    if self.log_scale_method == "split":
                        if self.scatter_mode == "split":
                    #if self.scatter_mode == "split":
                        #self.obj_positive.scene.camera.zoom(1.005)# if elevation_turn_bool==True else 0.995)
                        #self.obj_negative.scene.camera.zoom(1.005)# if elevation_turn_bool==True else 0.995)
                            self.obj_positive.scene.render()
                        #self.obj_negative.scene.render()
                else:
                    #self.obj.scene.camera.zoom(1.005)# if elevation_turn_bool==True else 0.995)
                    self.obj.scene.render()



                ###if mlab.view()[1] <=1: elevation_turn_bool = True
                ###if mlab.view()[1] >=179: elevation_turn_bool = False

                print(mlab.view()) 
                yield
        anim()


file = []
root = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/"
for i in range(1,51):
    #("tmp_%04d.png" % i)
    #file.append(root + "snap_%03d_delta_rho_fluid.h5" % i)
    file.append(root + "delta_rho_fluid_" + str(i) +  ".h5")
#print(file)
#sys.exit(0)

test = plot_class(file,indices=["singles",48,49],verbose = True)
test.symmetric_colorbar()

test.scatter(rescale_factor=1)
test.log_scale(method="split")
#test.save()
#test.move_camera()
#test.help_indexing()
test.mask(percentile=(5,95,"outside"),method="limits") # does this work with log_scale
#test.mask(percentage=20,method="rng")
#test.offscreen_rendering()
test.show()
#test.move_camera()
test.execute()
print("Done")