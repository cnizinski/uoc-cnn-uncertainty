import numpy as np
import math
import pandas as pd
from texture_pkg import interpolate


def unfold(img, snake):
    '''
    Unfolds grayscale image into spectrum
    Inputs  : img (grayscale as np array)
              snake (bool, snake unfold?)
    Outputs : spectrum
    Usage   : imspec = unfold(img=my_img, snake=True)
    '''
    # Check for grayscale image
    if len(img.shape) > 2:
        print("Input grayscale image")
        return
    # Append img rows to one another
    imh = img.shape[0]
    imw = img.shape[1]
    spec = np.zeros((imh*imw), dtype=np.uint8)
    for i in range(0,imh):
        curr_row = img[i]
        if snake is True and (i%2 == 1):
            spec[i*imw : i*imw+imw] = curr_row[::-1]
        else:
            spec[i*imw : i*imw+imw] = curr_row
    # Return unfolded image spectrum
    return spec


def get_lr(imgspec, cpt, scale):
    '''
    Return left/right points from center point
    Inputs  : imgspec (np array)
              center ((x,y) tuple of center point)
              scale (integer distance)
    Outputs : lpt ((x,y) tuple of left point)
              rpt ((x,y) tuple of right point)
    Usage   : l, r = get_lr(imspec, (50,150), 20)
    '''
    # Initialize variables
    cidx = cpt[0]
    lidx, ridx, = cidx, cidx
    ldist, rdist = 0, 0
    # Get course step size
    cstep = np.amax([1, int(scale/10)])
    # Coarse left sweep
    while (ldist < scale):
        lidx -= cstep
        ldx = lidx - cidx
        ldy = int(imgspec[lidx]) - int(imgspec[cidx])
        ldist = math.hypot(ldx, ldy)
    # Fine left sweep
    lx = lidx
    ly = imgspec[lidx]
    while (ldist > scale):
        lx += 0.004
        ly = interpolate(lx,(lidx,imgspec[lidx]),(lidx+1,imgspec[lidx+1]))
        ldx = lx - cidx
        ldy = ly - imgspec[cidx]
        ldist = math.hypot(ldx, ldy)
    lpt = (lx, ly)
    # Coarse right sweep
    while (rdist < scale):
        ridx += cstep
        rdx = ridx - cidx
        rdy = int(imgspec[ridx]) - int(imgspec[cidx])
        rdist = math.hypot(rdx, rdy)
    # Fine right sweep
    rx = ridx
    ry = imgspec[ridx]
    while (rdist > scale):
        rx -= 0.004
        ry = interpolate(rx,(ridx-1,imgspec[ridx-1]),(ridx,imgspec[ridx]))
        rdx = rx - cidx
        rdy = ry - imgspec[cidx]
        rdist = math.hypot(rdx, rdy)
    rpt = (rx, ry)
    # Return left and right point tuples
    return lpt, rpt


def calc_angle(ctup, ltup, rtup):
    '''
    Calculates angle
    Inputs  : ctup (center (x,y) tuple)
              ltup (left (x,y) tuple)
              rtup (right (x,y) tuple)
    Outputs : angle (radians)
    Usage   : angle = calc_angle((5,5), (3,7), (7,5))
    '''
    # Fongaro convention for AMT
    # Get vectors, unit vectors, -vecAB
    v1 = (-ltup[0] + float(ctup[0]), -ltup[1] + float(ctup[1]))
    v1u = v1 / np.linalg.norm(v1)
    # vecAC
    v2 = (rtup[0] - float(ctup[0]), rtup[1] - float(ctup[1]))
    v2u = v2 / np.linalg.norm(v2)
    # Get angle between -vecAB and vecAC
    vdot = v1u[0]*v2u[0] + v1u[1]*v2u[1]
    angle = np.arccos(np.clip(vdot,-1.0,1.0))
    #print(ltup, ctup, rtup, angle)
    return angle


def img_amt(img_arr, max_scale, n, snakes):
    '''
    Returns mean angle data for single image
    Inputs  : img_arr (np array of grayscale image)
              max_scale (int, pixels)
              n (number of samples or fraction of pixels)
              snakes (bool, snake on unfolding)
    Outputs : data_dict
    Usage   : my_dict = img_amt(img, 200, 0.03, snakes=True)
              my_dict = img_amt(img, 200, 1000, snakes=True)
    '''
    # Initialize output dictionary
    data_dict = {'Scale':[], 'MA':[]}
    # Unfold image
    ufspec = unfold(img_arr, snake=snakes)
    lenspec = len(ufspec)
    ufidxs = np.linspace(0, lenspec, num=lenspec+1, dtype='uint32')
    # Handle sampling procedure
    if n < 1.0:
        n_samples = int(lenspec * n)
    elif (n >= 1.0) and (n < 10000):
        n_samples = int(n)
    else:
        print("Invalid n. Enter a smaller number")
        return
    print("Measuring {0:4d} scales at {1:6d} samples per scale."\
        .format(max_scale, n_samples))
    # Iterate through scales
    for scale in range(1, max_scale+1):
        if scale % 50 == 0:
            print("Current scale = {0:4d}\r".format(scale))
        angs = []
        # Sample pixels and iterate through each
        ends = scale + 15
        tempidxs = ufidxs[ends:-ends]
        randidxs = np.random.choice(tempidxs,size=n_samples,replace=False)
        for idx in randidxs:
            curr_cpt = (idx, ufspec[idx])
            curr_lpt, curr_rpt = get_lr(ufspec, curr_cpt, scale)
            angs.append(calc_angle(curr_cpt, curr_lpt, curr_rpt))
        # Add mean results to output dict
        data_dict['Scale'].append(scale)
        data_dict['MA'].append(np.nanmean(angs))
    # Return data
    return data_dict