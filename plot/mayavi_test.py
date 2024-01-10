"""
import numpy as np
from mayavi import mlab
x, y = np.mgrid[0:3:1,0:3:1]
s = mlab.surf(x, y, np.asarray(x*0.1, 'd'))

@mlab.animate
def anim():
    for i in range(10):
        s.mlab_source.scalars = np.asarray(x*0.1*(i+1), 'd')
        yield

anim()
mlab.show()
"""

from mayavi import mlab
from tvtk.tools import visual
from vtk.util import colors as color
import numpy as np


@mlab.animate(delay = 100)
def updateAnimation():
    t = 0.0
    while True:
        ball.mlab_source.set(x = np.cos(t), y = np.sin(t), z = 0)
        t += 0.1
        yield

ball = mlab.points3d(np.array(1.), np.array(0.), np.array(0.))

updateAnimation()
mlab.show()
# Create a figure
#f = mlab.figure()#size=(200,200))
# Tell visual to use this as the viewer.
#visual.set_viewer(f)




# A silly visualization.
#mlab.test_plot3d()
"""
# Even sillier animation.
b1 = visual.box(x=0,color=color.blue)
b2 = visual.box(x=4., color=color.red)
b3 = visual.box(x=-4, color=color.red)
b1.v = 5.0

@mlab.show
@mlab.animate(delay=10)
def anim():
    """
#Animate the b1 box
"""
    while 1:
        b1.x = b1.x + b1.v*0.01
        if b1.x > 3 or b1.x < -3:
            b1.v = -b1.v
        yield

# Run the animation.
anim()
"""
