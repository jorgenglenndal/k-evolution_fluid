import numpy as np
import sys
import h5py
import copy
from mayavi import mlab




class visualization_class:
    def __init__(self,filename,filetype = "hdf5",indices="all",data_indexing="numpy",verbose=True):
        self.filetype = filetype
        self.verbose_bool = verbose
        if self.verbose_bool: print("Filetype is " + self.filetype + ".")
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
        
        if self.filetype == "hdf5" or self.filetype == "HDF5" or self.filetype == "h5":
            self.data = self.load_hdf5_data(0)
        elif self.filetype == "npy" or self.filetype == "numpy":
            self.data = self.load_npy_data(0)
        else:
            print("Unknown filetype. Aborting...")
            sys.exit(1)
        self.shape = np.shape(self.data) # all datasets should have the same shape
        
        if self.shape[0] == self.shape[1] == self.shape[2]:
            self.n_grid = self.shape[0] 
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
        self.rescale_all_data_bool = False
        #self.scatter_mode = "standard"
        #self.verbose_bool = False

        if len(self.shape) == 3:
            self.ScalarData = True 
            if self.verbose_bool: print("")
            if self.verbose_bool: print("Working with scalar data.")
            if self.verbose_bool: print("")
            
        elif len(self.shape) == 4:
            self.VectorData = True
            if self.verbose_bool: print("")
            if self.verbose_bool: print("Working with vector data")
            if self.verbose_bool: print("Vector data will be converted to scalar data by the Pythagorean norm.")
            #print("Indexing probably wrong...Aborting...")
            #sys.exit(1)
            #data_x = self.data[:,:,:,0]
            #data_y = self.data[:,:,:,1]
            #data_z = self.data[:,:,:,2]
            #self.data = np.sqrt(data_x**2 + data_y**2 + data_z**2) 
            self.shape = np.ones(3)*self.n_grid
            if self.verbose_bool: print("New shape is " +"("+str(int(self.shape[0]))+", "+str(int(self.shape[1]))+", "+str(int(self.shape[2]))+")")
            print("")
        else:
            print("Data shape not accepted. Aborting...")
            sys.exit(1)

    def load_npy_data(self,i,print_shape = True):
        data = np.load(self.filename[i])
        if print_shape and self.verbose_bool:
            print("Shape of " + self.filename[i] +" = " + str(np.shape(data)))
        return data
        
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
            #print(data)
            #print(type(file[keys[0]])) 
        if self.verbose_bool and print_shape:
            print("Shape of " + self.filename[i] +" = " +str(np.shape(data)))
        return data
    
    def mask(self,method = "rng",percent=0.5,percentile=(0.25,99.75,"outside"),seed = 1234):
        self.mask_bool = True
        self.mask_method = method
        if method =="rng":
            self.mask_seed = seed
            self.mask_percent = percent
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
    def rescale_all_data(self, rescale_all_data_factor):
        self.rescale_all_data_bool = True
        if self.verbose_bool: print("Rescaling selected.")
        self.rescale_all_data_factor = rescale_all_data_factor
        if isinstance(self.rescale_all_data_factor, list) == False:
            if self.verbose_bool: print("All data is rescaled by "+ str(self.rescale_all_data_factor))
            self.rescale_all_data_factor = [self.rescale_all_data_factor]



    def mask_func(self):
        if self.verbose_bool: print("Masking...")
        
        if self.mask_method == "rng":
            if self.verbose_bool: print("Keeping "+ str(self.mask_percent)+"% " "of the data")
            if self.mask_seed == False:
                if self.verbose_bool: print("Using 'numpy.random.default_rng()' for uniform random sampling")
                rng = np.random.default_rng()
            else:
                if self.verbose_bool: print("Using 'numpy.random.default_rng(seed = "+ str(self.mask_seed) +")' for uniform random sampling.")
                rng = np.random.default_rng(self.mask_seed)
                
            random = rng.random(size=self.n_grid**3)   
            random = random.reshape(self.shape)
            self.condition = random >= (100-self.mask_percent)/100
            actual_percent = len((self.condition[self.condition]).flatten())/self.n_grid**3*100
            if self.verbose_bool: print(f"Percent of data kept: {actual_percent:.3f}")
            self.rng_generated_bool = True
        
        elif self.mask_method== "limits":
            #if self.VectorData: print("method: 'limits' does not work on vector data. Aborting..."); sys.exit(1)
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
            actual_percent = len((self.condition[self.condition]).flatten())/self.n_grid**3*100
            if self.verbose_bool: print(f"Percent of data kept: {actual_percent:.3f}")           

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
        
    # defines vmin and vmax
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
                if self.max_positive >= self.max_negative:
                    self.vmax = self.max_positive
                    
                else:
                    self.vmax = self.max_negative
                
                if self.min_positive <= self.min_negative:
                    self.vmin = self.min_positive
                else:
                    self.vmin = self.min_negative

        else:
            flat_data = self.data.flatten()
            pos_loop = 0
            neg_loop = 0
            for index in range(len(flat_data)):
                if flat_data[index] > 0:
                    pos_loop +=1
                elif flat_data[index] == 0:
                    continue
                else:
                    neg_loop +=1
            if pos_loop == 0:
                if self.verbose_bool: print("Only negative values. Symmetric colorbar has been auto-deselected for this file.")
                self.symmetric_colorbar_bool = False
                #if self.verbose_bool: print("Symmetric colorbar has been auto-deselected for this file.")
            if neg_loop == 0:
                if self.verbose_bool: print("Only positive values. Symmetric colorbar has been auto-deselected for this file.")
                self.symmetric_colorbar_bool = False
            
            self.vmin = - self.abs_max
            self.vmax = self.abs_max
    """     
    def plot_threeD_quiver(self,upscale_factor):
        #mlab.clf()
        obj = mlab.quiver3d(self.xx, self.yy, self.zz, self.data_x, self.data_y, self.data_z,scale_mode="vector",scale_factor = 1/self.n_grid*upscale_factor,mode = "arrow")#,colormap = "inferno")
        #mlab.vectorbar(object=obj,orientation='vertical')
        mlab.colorbar(orientation='vertical')

        obj.glyph.glyph.clamping = False
        #obj.module_manager.vector_lut_manager.vector_bar.orientation = 'vertical'
        mlab.outline(obj)
        self.plot = True
    """
        
    def scatter_func(self):
        if self.verbose_bool:
            print("")
            print("Plotting scatter...")
            print("")

        scale_factor = self.rescale_factor/(self.abs_max*self.n_grid) # normalizing such that the largest glyph size is independent of the corresponding scalar value and the grid size
        #else:
        #    scale_factor = self.rescale_factor/(self.n_grid - 1)

        #if self.symmetric_colorbar_bool:
        #    colormap = 'seismic'
            
        "I was not able to implement 'colorcet' in 'mayavi'."
            #import colorcet as cc
            #from matplotlib.cm import get_cmap

            #colormap = str(get_cmap("cet_fire"))
            #colormap = cc.m_rainbow4
        #elif self.log_scale_bool:
        #    if self.log_scale_method == "abs":
        #        colormap = 'YlOrBr'

        #else:
        #    colormap = 'jet'

        if self.log_scale_bool:
            if self.log_scale_method == "abs":
                colormap = 'hot'
                self.obj = mlab.points3d(self.xx_current,self.yy_current,self.zz_current,self.data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor)#,vmin = self.vmin, vmax = self.vmax)#,extent=[0, 1, 0, 2, 0, 3])
                self.obj.glyph.glyph.clamping = False
                self.obj.actor.property.representation = "wireframe" # color bug occurs for option: "surface"
                cb = mlab.scalarbar(object=self.obj,orientation='vertical')#,title= "overdensity for velocity")
                self.obj.module_manager.scalar_lut_manager.reverse_lut = True
                cb.scalar_bar.unconstrained_font_size = True
                cb.label_text_property.font_size=30
                mlab.outline(self.obj,extent = [0,1,0,1,0,1])
                self.obj.module_manager.scalar_lut_manager.scalar_bar_representation.show_border = 'off'

            elif self.log_scale_method == "split":
                if self.scatter_mode == "only positive":
                    colormap = "Reds"
                    self.obj = mlab.points3d(self.xx_positive,self.yy_positive,self.zz_positive,self.positive_data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor)#,vmin = self.vmin, vmax = self.vmax)#,extent=[0, 1, 0, 2, 0, 3])
                    self.obj.glyph.glyph.clamping = False
                    self.obj.actor.property.representation = "wireframe" # color bug occurs for option: "surface"
                    cb = mlab.scalarbar(object=self.obj,orientation='vertical',title= "+")
                    self.obj.module_manager.scalar_lut_manager.title_text_property.bold = False
                    self.obj.module_manager.scalar_lut_manager.title_text_property.italic = False
                    self.obj.module_manager.scalar_lut_manager.title_text_property.use_tight_bounding_box = True
                    mlab.outline(self.obj,extent = [0,1,0,1,0,1])
                    cb.scalar_bar.unconstrained_font_size = True
                    cb.label_text_property.font_size=30
                    self.obj.module_manager.scalar_lut_manager.scalar_bar_representation.show_border = 'off'
                    self.obj.module_manager.scalar_lut_manager.title_text_property.font_size = 100

                elif self.scatter_mode == "only negative":
                    colormap = "Blues"
                    self.obj = mlab.points3d(self.xx_negative,self.yy_negative,self.zz_negative,self.negative_data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor)#,extent=[0, 1, 0, 1, 0, 1])
                    self.obj.glyph.glyph.clamping = False
                    self.obj.actor.property.representation = "wireframe" # color bug occurs for option: "surface"
                    cb = mlab.scalarbar(object=self.obj,orientation='vertical',title= "-")
                    self.obj.module_manager.scalar_lut_manager.title_text_property.bold = False
                    self.obj.module_manager.scalar_lut_manager.title_text_property.italic = False
                    self.obj.module_manager.scalar_lut_manager.title_text_property.use_tight_bounding_box = True
                    mlab.outline(self.obj,extent = [0,1,0,1,0,1])
                    cb.scalar_bar.unconstrained_font_size = True
                    cb.label_text_property.font_size=30
                    self.obj.module_manager.scalar_lut_manager.scalar_bar_representation.show_border = 'off'
                    self.obj.module_manager.scalar_lut_manager.title_text_property.font_size = 100

                elif self.scatter_mode == "split":
                    colormap = "Reds"
                    self.obj_positive = mlab.points3d(self.xx_positive,self.yy_positive,self.zz_positive,self.positive_data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor,vmin = self.vmin if self.symmetric_colorbar_bool else None, vmax = self.vmax if self.symmetric_colorbar_bool else None)#,extent=[0, 1, 0, 1, 0, 1])
                    self.obj_positive.glyph.glyph.clamping = False
                    self.obj_positive.actor.property.representation = "wireframe" # color bug occurs for option: "surface"
                    cb_pos = mlab.scalarbar(object=self.obj_positive,orientation='vertical',title = "+")
                    mlab.outline(self.obj_positive,extent = [0,1,0,1,0,1])
                    cb_pos.scalar_bar.unconstrained_font_size = True
                    cb_pos.label_text_property.font_size=30
                    self.obj_positive.module_manager.scalar_lut_manager.title_text_property.font_size = 100
                    #self.obj_positive.module_manager.scalar_lut_manager.scalar_bar.draw_annotations = False
                    self.obj_positive.module_manager.scalar_lut_manager.scalar_bar_representation.show_border = 'off'



                    colormap = "Blues"
                    self.obj_negative = mlab.points3d(self.xx_negative,self.yy_negative,self.zz_negative,self.negative_data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor,vmin = self.vmin if self.symmetric_colorbar_bool else None, vmax = self.vmax if self.symmetric_colorbar_bool else None)#,extent=[0, 1, 0, 1, 0, 1])
                    self.obj_negative.glyph.glyph.clamping = False
                    self.obj_negative.actor.property.representation = "wireframe" # color bug occurs for option: "surface"
                    cb_neg = mlab.scalarbar(object=self.obj_negative,orientation='vertical',title = "-")
                    cb_neg.scalar_bar.unconstrained_font_size = True
                    cb_neg.label_text_property.font_size=30
                    self.obj_negative.module_manager.scalar_lut_manager.scalar_bar.text_position = 'precede_scalar_bar'
                    self.obj_negative.module_manager.scalar_lut_manager.title_text_property.font_size = 100
                    #self.obj_negative.module_manager.scalar_lut_manager.scalar_bar.draw_annotations = False
                    self.obj_negative.module_manager.scalar_lut_manager.scalar_bar_representation.show_border = 'off'

                    #mlab.outline(self.obj_negative)

                    self.obj_positive.module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([0.01,  0.1])
                    self.obj_negative.module_manager.scalar_lut_manager.scalar_bar_representation.position = np.array([0.89,  0.1])
                    
                    self.obj_positive.module_manager.scalar_lut_manager.title_text_property.bold = False
                    self.obj_positive.module_manager.scalar_lut_manager.title_text_property.italic = False
                    self.obj_positive.module_manager.scalar_lut_manager.title_text_property.use_tight_bounding_box = True

                    self.obj_negative.module_manager.scalar_lut_manager.title_text_property.bold = False
                    self.obj_negative.module_manager.scalar_lut_manager.title_text_property.italic = False
                    self.obj_negative.module_manager.scalar_lut_manager.title_text_property.use_tight_bounding_box = True

                else:
                    print("Error in scatter mode. Aborting...")
                    sys.exit(1)


            else:
                print("Unknown log scale method. Aborting...")
                sys.exit(1)

        else:
            colormap = 'RdBu' if self.symmetric_colorbar_bool else 'jet'
            self.obj = mlab.points3d(self.xx_current,self.yy_current,self.zz_current,self.data,mode = 'sphere',transparent = False,resolution=8,scale_mode='scalar',opacity=1,colormap = colormap,scale_factor=scale_factor,vmin = self.vmin if self.symmetric_colorbar_bool else None, vmax = self.vmax if self.symmetric_colorbar_bool else None)
            self.obj.glyph.glyph.clamping = False 
            self.obj.actor.property.representation = "wireframe" # color bug occurs for option: "surface"
            cb = mlab.colorbar(object=self.obj,orientation='vertical')#,title= "Overdensity")
            self.obj.module_manager.scalar_lut_manager.reverse_lut = True if self.symmetric_colorbar_bool else False
            mlab.outline(self.obj,extent = [0,1,0,1,0,1])
            cb.scalar_bar.unconstrained_font_size = True
            cb.label_text_property.font_size=30
            self.obj.module_manager.scalar_lut_manager.scalar_bar_representation.show_border = 'off'


            #mlab.title("Divergence",height=0.88)
            #mlab.title.text.y_position=(0.88)
            
            
            #mlab.actor.minimum_size = np.array([10, 10])
            #mlab.actor.position = np.array([0.225, 0.876])
            #mlab.actor.position2 = np.array([0.55, 1.  ])
            #mlab.actor.position = np.array([0.225, 0.876])
            #mlab.y_position = 0.876
        
        #mlab.xlabel('X=Y=Z [300 Mpc/h]')
        #mlab.ylabel('y [300 Mpc/h]')
        #mlab.zlabel('z [300 Mpc/h]')
        #mlab.view(distance='auto', focalpoint=[0, 0, 0])
        #mlab.view(azimuth=270)
        #mlab.view(azimuth=0, elevation=90)
        self.plot_bool = True
    
    def plotting_loop_func(self):
        def inside_loop_func(i):
            mlab.figure(size=(1200, 1080))
            if self.filetype == "hdf5" or self.filetype == "HDF5" or self.filetype == "h5":
                self.data = self.load_hdf5_data(i,print_shape=False) # reading all the data
            elif self.filetype == "npy" or self.filetype == "numpy":
                self.data = self.load_npy_data(i,print_shape=False) # reading all the data
            else:
                print("Bad filetype. Aborting...")
                sys.exit(1)

            if self.VectorData:
                self.data_x = self.data[:,:,:,0]
                self.data_y = self.data[:,:,:,1]
                self.data_z = self.data[:,:,:,2]
                #self.data = np.sqrt(data_x**2 + data_y**2 + data_z**2)
            
            if self.indexing == "xyz" or self.indexing == "npy" or self.indexing == "numpy":
                if self.ScalarData: self.data = self.data.transpose((2, 1, 0)) # transposing to get ijk indexing
                if self.VectorData:
                    self.data_x= self.data_x.transpose((2, 1, 0)) # transposing to get ijk indexing
                    self.data_y= self.data_y.transpose((2, 1, 0))
                    self.data_z= self.data_z.transpose((2, 1, 0))
            else:
                if self.indexing != "ijk":
                    print("Invalid 'data_indexing'. Aborting...")
                    sys.exit(1) 

            if self.VectorData: self.data = np.sqrt(self.data_x**2 + self.data_y**2 + self.data_z**2)


            # rescaling all data by user specified factor (scalar)
            if self.rescale_all_data_bool:
                if len(self.rescale_all_data_factor)==1:
                    self.data *= self.rescale_all_data_factor[0]
                    if self.verbose_bool: print("Data has been rescaled, by user, by a factor " + str(self.rescale_all_data_factor[0]))
                else:
                    self.data *= self.rescale_all_data_factor[i]
                    if self.verbose_bool: print("Data has been rescaled, by user, by a factor " + str(self.rescale_all_data_factor[i]))
                
            
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
                        self.xx_current, self.yy_current, self.zz_current = self.xx[self.condition], self.yy[self.condition], self.zz[self.condition]
                else:
                    print("Unknown mask method. Aborting...")
                    sys.exit(1)
            else:
                self.xx_current, self.yy_current, self.zz_current =  copy.deepcopy(self.xx),copy.deepcopy(self.yy),copy.deepcopy(self.zz) 
                
            # data and coords have been masked
            if self.log_scale_bool:
                self.rescale_data_factor = 1
                self.rescale_data_factor_positive = 1
                self.rescale_data_factor_negative = 1
                if self.log_scale_method == "abs":
                    self.data = np.abs(self.data)
                    if np.nanmin(self.data) == 0:
                        condition = self.data > 0
                        self.data = self.data[condition] # to avoid log zero
                        self.xx_current, self.yy_current, self.zz_current = self.xx_current[condition], self.yy_current[condition], self.zz_current[condition]
                    data_test = self.data[self.data < 1.1]  
                    if len(data_test.flatten()) > 0:
                        data_min = np.nanmin(data_test)
                        self.rescale_data_factor = 1.1/data_min 
                        self.data *= self.rescale_data_factor
                        print("Warning: Data has been rescaled by a factor " + str(self.rescale_data_factor))
                    self.data = np.log10(self.data)
                    self.abs_max = np.nanmax(self.data)
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
                        # checking if data is smaller than one. Avoiding negative values after taking logarithm to ensure correct scaling.
                        data_test = self.negative_data[self.negative_data < 1.1]
                        if len(data_test.flatten()) > 0:
                            data_min = np.nanmin(data_test)
                            self.rescale_data_factor_negative = 1.1/data_min
                            self.negative_data *= self.rescale_data_factor_negative
                            print("Warning: Data has been rescaled by a factor " + str(self.rescale_data_factor_negative))
                        self.negative_data = np.log10(self.negative_data)
                        self.abs_max = np.nanmax(self.negative_data)
                    elif len(self.negative_data) == 0:
                        self.scatter_mode = "only positive"
                        print("no negative")
                        self.xx_positive = self.xx_current[positive_booleans]
                        self.yy_positive = self.yy_current[positive_booleans]
                        self.zz_positive = self.zz_current[positive_booleans]
                        data_test = self.positive_data[self.positive_data < 1.1]
                        if len(data_test.flatten()) > 0:
                            data_min = np.nanmin(data_test)
                            self.rescale_data_factor_positive = 1.1/data_min
                            self.positive_data *= self.rescale_data_factor_positive
                            print("Warning: Data has been rescaled by a factor " + str(self.rescale_data_factor_positive))
                        self.positive_data = np.log10(self.positive_data)
                        self.abs_max = np.nanmax(self.positive_data)
                    else:
                        self.scatter_mode = "split"

                        self.xx_positive = self.xx_current[positive_booleans]
                        self.yy_positive = self.yy_current[positive_booleans]
                        self.zz_positive = self.zz_current[positive_booleans]
                        self.xx_negative = self.xx_current[negative_booleans]
                        self.yy_negative = self.yy_current[negative_booleans]
                        self.zz_negative = self.zz_current[negative_booleans]


                        data_test = self.positive_data[self.positive_data < 1.1] 
                        if len(data_test.flatten()) > 0:
                            data_min = np.nanmin(data_test)
                            self.rescale_data_factor_positive = 1.1/data_min 
                            

                        data_test = self.negative_data[self.negative_data < 1.1] 
                        if len(data_test.flatten()) > 0:
                            data_min = np.nanmin(data_test)
                            self.rescale_data_factor_negative = 1.1/data_min 
                        
                        if self.rescale_data_factor_positive > 1 or self.rescale_data_factor_negative > 1:
                            self.rescale_data_factor = np.nanmax([self.rescale_data_factor_positive,self.rescale_data_factor_negative])    
                            self.positive_data *= self.rescale_data_factor
                            self.negative_data *= self.rescale_data_factor
                            print("Warning: Data has been rescaled by a factor " + str(self.rescale_data_factor))
                        
                        self.positive_data = np.log10(self.positive_data)
                        self.negative_data = np.log10(self.negative_data)

                        self.max_positive = np.nanmax(self.positive_data)
                        self.max_negative = np.nanmax(self.negative_data)
                        self.min_positive = np.nanmin(self.positive_data)
                        self.min_negative = np.nanmin(self.negative_data)

                        self.abs_max = self.max_positive if self.max_positive >= self.max_negative else self.max_negative
                        if self.symmetric_colorbar_bool == False:
                            if self.verbose_bool: print("")
                            if self.verbose_bool: print("Symmetric colorbar is recommended to get more consistent colors. Scaling is always consistent.")
                            if self.verbose_bool: print("")
                    
                else:
                    print("Unknown log_scale method. Available methods are: 'abs', 'split'. Aborting...")
                    sys.exit(1)
            else:
                self.abs_max = np.nanmax(np.abs(self.data))

            if self.symmetric_colorbar_bool:
                self.symmetric_colorbar_func()
            
            if self.scatter_bool:
                if self.mask_bool == False and self.verbose_bool: print("Plotting without masking... May be very slow.")
                self.scatter_func()
                #mlab.view(azimuth=None, elevation=None, distance=None, focalpoint=None,roll=None, reset_roll=True, figure=None)
                #mlab.view(146.2499999999997, 48.48561031724544, 3.9999999999999947, np.array([0.50011478, 0.50011478, 0.50011774]))
                ###69.04300699344203
                if self.move_camera_bool == False: mlab.view(-119.00051786825493, 69.04300699344203, 3, np.array([0.5,0.45,0.4]))#[0.50011478, 0.50011478, 0.50011774]))
            #mlab.savefig("divergence_mayavi_061053.png")    
            #mlab.show()
            if self.plot_bool:
                #if self.mask_bool == False and self.verbose_bool: print("Plotting without masking... May be very slow.")
                if self.save_bool:
                    if int(len(self.filename)) > 10001:
                        print("Too many files. The mlab.savefig() line must be changed from 04d to e.g. 05d. Aborting...")
                        sys.exit(1)
                    if self.verbose_bool: print("Saving files as ./tmp_i.png")
                    #mlab.jpeg_quality = 100
                    mlab.savefig("tmp_%04d.png" % i,size=(28,28))
                if self.move_camera_bool:
                    self.move_camera_func()
                if self.show_bool: 
                    mlab.show()
                else:
                    mlab.close()
            else:
                if self.verbose_bool: print("Nothing has been plotted.")
            
        #import copy 
        # defining coordinate matrices independent of self.xx, ...    
        #self.xx_current, self.yy_current, self.zz_current =  copy.deepcopy(self.xx),copy.deepcopy(self.yy),copy.deepcopy(self.zz)
        
        if self.offscreen_rendering_bool:
            mlab.options.offscreen = True
            self.show_bool = False
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
            #import inspect

            #line_number = inspect.currentframe().f_back.f_lineno
            print("'indices' invalid. Aborting...")
            #print("Line number:", line_number)
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
            self.step = int(self.indices[3]) if len(self.indices) >= 4 else 1
            self.plotting_loop_func()

        else:
            print("'indices' set incorrectly. Aborting...")
            sys.exit(1)    
        print("")                    
    
    #def help_indexing(self):
    #    print("")
    #    #print("Help: Standard numpy indexing is called 'xyz'. A numpy array, arr, of shape (n_z, n_y, n_x) will have indices x, y and z such that arr[z,y,x] will return the value of the element located at index z in the z-direction, index y in the y-direction and index x in the x-direction. Mayavi uses 'ijk' indexing, i.e., arr[i,j,k] returns the value of the element located at index i in the x-direction, index j in the y-direction and index k in the z-direction. If the data array, read from the input file, uses 'xyz' indexing, the code will rearange the data correctly if, and only if, the parameter 'indexing' is set to its default value: 'xyz'. Regarding gevolution: gevolution uses 'ijk' indexing internally, but when the outputted HDF5 files are read in python, the data is in 'xyz' indexing. End help.")
    #    print("Help: For both hdf5 files and npy files the 'indexing' should be set to 'numpy'. This is because the hdf5 files are loaded as numpy arrays. However, if your hdf5 files were not produced by the gevolution code, you might want to check that the mayavi plots look correct in the x and z direction (probably not necessary). This can be done by making a test hdf5 file with zero everywhere except in the x-y plane at z=0 (so you must know the internal indexing in the code you are using). If you get the correct mayavi plot the indexing is correct. If the plane lies in the z-y plane at x=0 you just have to set the 'indexing' variable to 'ijk'. End help.")
    #          
    #          
    #          #If you implement some other filetype, you must check if the indexing is as in numpy or if it is 'ijk'.")
    #          
    #          #This applies at least to the hdf5 files from gevolution (I've checked), even though gevolution does not use numpy indexing internally. If your hdf5 file was not produced by gevolution you should, probably, still use numpy indexing, because I have only used the standardloading method for the hdf5 files.")
    #    print("")
        
    #def help_method(self):
    #    print("")
    #    print("Help: Possible options for 'method' is 'rng' and 'limits'. 'rng' needs the parameter 'percent' to determine how many percent of the data to keep. 'limits' needs a parameter 'percentile' on the form [a,b,c], where a is the bottom percentile, of interest, of the data and b is the top percentile, of interest, of the data. c must be either 'outside' or 'inside' the limits. Limits are excluded from the resulting dataset. End help. ")
    #    print("")

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



"""

file = []
root = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test/"
for i in range(1,51):
    #("tmp_%04d.png" % i)
    #file.append(root + "snap_%03d_delta_rho_fluid.h5" % i)
    file.append(root + "delta_rho_fluid_" + str(i) +  ".h5")
#print(file)
#sys.exit(0)


divergence ="/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test4/" +"snap_000_div_v_upper_fluid.h5"
overdensity ="/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/test4/"+ "snap_000_delta_rho_fluid.h5"
#test_file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/new_update_funcs/" + "phi_old_test.h5"
new_div = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/140601/"  + "snap_000_div_v_upper_fluid.h5"#"div_v_upper_fluid_1.h5"
new_density = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/140601/"  + "snap_000_delta_rho_fluid.h5"#"div_v_upper_fluid_1.h5"

test_file = "/mn/stornext/d5/data/jorgeagl/kevolution_output/test/tests/remove/hiclass_tests/test1/snap_001_v_upper_i_fluid.h5"
#N = 25
#A = np.zeros((N,N,N))
#for k in range(N):
#    for j in range(N):
#        for i in range(N):
#            A[k,j,i] = i**5-j**5+k**5
#
#np.save("A_test",A)


#test = visualization_class("A_test.npy",filetype="npy")# 
#test = visualization_class(filename=divergence)# ,indices=["singles",0])
test = visualization_class(filename=test_file)#
test.rescale_all_data(1000000000)

##test = plot_class(test_file)
#test = plot_class(file,indices=["singles",49])
test.symmetric_colorbar()

test.scatter(rescale_factor=1)
#test.log_scale(method="split")
#test.save()
#test.move_camera()
#test.help_indexing()
test.mask(percentile=(0.5,99.9,"outside"),method="limits") 
#test.mask(percent=2,method="rng")
#test.offscreen_rendering()
test.show()
#test.move_camera()
test.execute()
#print(test.rescale_data_factor_positive)
print("Done")
"""