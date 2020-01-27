import numpy as np
import math


def interpolate(xi, p0, p1):
    '''
    Numerically interpolates between (x0,y0) and (x1,y1)
    Inputs  : interp xi, known x's (x0, x1)
              known y's (y0, y1)
    Outputs : unknown yi
    '''
    # Interpolate
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]) ,float(p1[1])
    yi = y0 + (y1-y0) * (float(xi)-x0)/(x1-x0)
    return yi