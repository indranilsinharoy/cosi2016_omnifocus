# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:          omnifocuslib.py
# Purpose:       helper functions for simulation of omnifocus image synthesis
#                using Zemax and PyZDDE. In particular, the results presented in
#                the paper: "Omnifocus image synthesis using lens swivel,"
#                I. Sinharoy, P. Rangarajan, and M. Christensen,  in Imaging and 
#                Applied Optics 2016, OSA. 
#                
# Author:        Indranil Sinharoy, Southern Methodist university, Dallas, TX.
#
# Copyright:     (c) Indranil Sinharoy 2015, 2016
# License:       MIT License
#-------------------------------------------------------------------------------
'''utility functions for simulation of omnifocus image synthesis using Zemax and 
   PyZDDE. Several functions are included for geometric optics computations, 
   automating the simulation, storing and retrieving image stack data into and 
   from tagged HDF5 format, etc. 
   Assumed wavelength for tracing rays in Zemax is 0.55 mu   
'''
from __future__ import division, print_function
import os 
import sys
import numpy as np
import cv2
import collections as co
import pyzdde.zdde as pyz
import pyzdde.arraytrace as at 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colorbar import make_axes
from IPython.core import display
import ipywidgets as widgets
from ipywidgets import interactive, interact, fixed  
import h5py as hdf
import time as time 
from scipy.misc import imsave 

# global variable
_initTime = 0.0

#%% File, directory management and other housekeeping utility functions
def get_directory_path(dirbranch=None):
    '''returns absolute path of the leaf directory 
    
    If directory branch is not present, the function creates the directories under 
    current file directory
    
    Parameters
    ----------
    dirbranch : tuple or None
        tuple of strings representing the directory branch. If `None`
        the current directory is returned
        
    Returns
    -------
    dirpath : string
        absolute path to the directory
    
    Example
    -------
    >>> get_directory_path(['data', 'imgstack'])
    C:\Projects\project_edof\data\imgstack
    
    '''
    wdir = os.path.dirname(os.path.realpath(__file__))
    if dirbranch:
        dirpath = os.path.join(wdir, *dirbranch)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        return dirpath
    else:
        return wdir

def set_hdf5_attribs(hdf5Obj, attribDict):
    """helper function to set attributes of h5py objects  

    Parameters
    ---------- 
    hdf5Obj : HDF5 object 
        group or dataset object (including an hdf5 file)
    attribDict : dict 
        attribute dict 
    """
    for key, value in attribDict.items():
        hdf5Obj.attrs[key] = value

def save_to_IMAdir(arr, imagename):
    """helper function to save a numpy ndarray as image into the Zemax's Image 
    directory 

    Parameters
    ---------- 
    arr : ndarray
        ndarray 
    imagename : string 
        image filename, e.g. 'gridofdots.png'

    Returns
    ------- 
    None
    """
    usr = os.path.expandvars("%userprofile%")
    IMAdir = os.path.join(usr, 'Documents\Zemax\IMAFiles')
    filename = os.path.join(IMAdir, imagename)
    imsave(name=filename, arr=arr)

def get_time(startClock=False):
    """returns elapsed time  

    The first time this function is called, `startClock` should be True.
    The function doesn't print the time for the initiating call. 

    Returns
    ------- 
    time : string 
        elapsed time from the previous call 

    """
    global _initTime
    if startClock:
        _initTime = time.clock()
    else:
        _cputime = time.clock() - _initTime
        m, s = divmod(_cputime, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

#%% Helper functions for plotting data

def show_pixel_grid(ax, pixX, pixY, gAlpha=None):
    """plots pixel grid on the given axes 
    
    Parameters
    ----------
    ax : axes object 
        figure axes object 
    pixX : integer
        number of pixels along x/columns
    pixY : integer 
        number of pixels along y/rows
    gAlpha : float (0 < gAlpha <= 1)
        alpha for the pixel grid
        
    Returns
    -------
    None
    """
    gAlpha = gAlpha if gAlpha is not None else 1.0
    xl = np.arange(1, pixX) - 0.5
    yl = np.arange(1, pixY) - 0.5
    ax.xaxis.set_minor_locator(plt.FixedLocator(xl))
    ax.yaxis.set_minor_locator(plt.FixedLocator(yl))
    ax.xaxis.set_tick_params(which='minor', length=0)
    ax.yaxis.set_tick_params(which='minor', length=0)
    ax.xaxis.set_tick_params(which='major', direction='out')
    ax.yaxis.set_tick_params(which='major', direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(which='minor', color='r', linestyle='-', linewidth=0.75, 
            alpha=gAlpha)

def show_around(img_data, pixX=20, pixY=None, ctrX=None, ctrY=None, 
                pixSize=None, pixGrid=False, retData=False, ax=None):
    """display pixX x pixY pixels around (ctrX, ctrY). 
    
    Parameters
    ----------
    img_data : list of list
        data returned by ln.zGetImageSimulation() 
    pixX : integer 
        number of pixels along x
    pixY : integer, optional
        number of pxiels along y. If `None` or 0, `pixY` is assumed equal
        to `pixX` 
    ctrX : integer, optional
        center pixel along x
    ctrY : integer, optional 
        center pixel  along y
    pixSize : real, optional
        length of the side of a square pixel.
        if passed, a second axes is shown to show the physical dimension 
    pixGrid : bool, optional
        if True, show pixel grid
    retData : bool, optional
        if True, the cropped data is returned
        and not plotted
    ax : axes object, optional
        if an axes object is passed the plot will be rendered 
        in the given axes and `plt.show()` will have to be invoked 
        by the caller. Use `0` if you want to use this function 
        with ipywidgets' interactive function.  

    
    Assumptions
    ------------
    pixX, pixY, ctrX, ctrY are not 0
    """
    img = np.array(img_data, dtype='uint8')
    M, N = img.shape[:2]
    ctrX, ctrY = ctrX or N/2, ctrY or M/2
    pixX = pixX or 20
    pixY = pixY if pixY else pixX
    sRow, eRow = int(ctrY - pixY/2), int(ctrY + pixY/2)
    sCol, eCol = int(ctrX - pixX/2), int(ctrX + pixX/2)
    # ensure bounds are right
    sRow = sRow if sRow >= 0 else 0
    eRow = eRow if eRow <= M else M
    sCol = sCol if sCol >= 0 else 0
    eCol = eCol if eCol <= N else N
    data = img[sRow:eRow, sCol:eCol, :]
    if retData:
        return data
    else:
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
        iAlpha = 0.8 if pixGrid else 1.0
        ax.imshow(data, interpolation='none', zorder=15, alpha=iAlpha)
        ax.set_xlabel('pixels', fontsize=10) 
        ax.set_ylabel('pixels', fontsize=10)
        if pixGrid:
            show_pixel_grid(ax=ax, pixX=pixX, pixY=pixY)
        if pixSize:
            # create a secondary axes
            ax2 = ax.twiny()
            ax3 = ax2.twinx()
            wideX = pixSize*(eCol-sCol)
            wideY = pixSize*(eRow-sRow)
            ax.xaxis.set_ticks_position('bottom') # required
            ax.yaxis.set_ticks_position('left')   # required
            ax2.set_xlim(0, wideX)
            ax3.set_ylim(wideY, 0)
            ax2.set_xlabel('phy. dim.', fontsize=10) 
            ax3.set_ylabel('phy. dim.', fontsize=10)
            ws = 'Width: {:2.3f}'.format(wideX)
            hs = 'Height: {:2.3f}'.format(wideY)
            ax.text(x=0.99, y=0.055, s=ws, transform = ax.transAxes, 
                    color='w', zorder=16, alpha=0.65, horizontalalignment='right')
            ax.text(x=0.99, y=0.02, s=hs, transform = ax.transAxes, 
                    color='w', zorder=16, alpha=0.65, horizontalalignment='right')
        if not ax:
            ax.set_aspect('equal')
            plt.show()

def imshow(image, fig=None, axes=None, subplot=None, interpol=None,
           xlabel=None, ylabel=None, figsize=None, cmap=None):
    """Rudimentary image display routine, for quick display of images without
    the spines and ticks 
    """
    if (subplot == None):
        subplot = int(111)
    if(fig==None):
        if figsize and isinstance(figsize, tuple) and len(figsize)==2:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        axes = fig.add_subplot(subplot)
    elif(axes==None):
        axes = fig.add_subplot(subplot)
    
    # plot the image
    if len(image.shape) > 2:
        imPtHandle = plt.imshow(image, interpolation=interpol)
    else:
        cmap = cmap if cmap is not None else cm.gray
        imPtHandle = plt.imshow(image, cmap=cmap, interpolation=interpol)
        
    # get the image height and width to set the axis limits
    try:
        pix_height, pix_width = image.shape
    except:
        pix_height, pix_width, _ = image.shape
    # Set the xlim and ylim to constrain the plot
    axes.set_xlim(0, pix_width-1)
    axes.set_ylim(pix_height-1, 0)
    # Set the xlabel and ylable if provided
    if(xlabel != None):
        axes.set_xlabel(xlabel)
    if(ylabel != None):
        axes.set_ylabel(ylabel)
    # Make the ticks to empty list
    axes.xaxis.set_ticks([])
    axes.yaxis.set_ticks([])
    return imPtHandle, fig, axes

def get_imlist(filePath, itype='jpeg'):
    """returns a list of filenames for all images of specified type in a
    directory

    Parameters
    ----------
    filePath : string
        full path name of the directory to be searched
    itype : string, optional
        type of images to be searched, for example -- 'jpeg', 'tiff', 'png',
        'dng', 'bmp' (without the dot(.))

    Returns
    -------
    imageFiles : list of strings
        list of image filenames with full path.
    """
    imlist = []
    opJoin = os.path.join
    dirList = os.listdir(filePath)
    if itype in ['jpeg', 'jpg']:
        extensions = ['.jpg', '.jpeg', '.jpe',]
    elif itype in ['tiff', 'tif']:
        extensions = ['.tiff', '.tif']
    else:
        extensions = [''.join(['.', itype.lower()]), ]
    for ext in extensions:
        imlist += [opJoin(filePath, f) for f in dirList if f.lower().endswith(ext)]
    return imlist

#%% Basic geometric-optics functions
def gaussian_lens_formula(u=None, v=None, f=None, infinity=10e20):
    """return the third value of the Gaussian lens formula, given any two

    Parameters
    ----------
    u : float, optional
        object distance from first principal plane. 
    v : float, optional
        image distance from rear principal plane 
    f : float, optional
        focal length
    infinity : float
        numerical value to represent infinity (default=10e20)

    Returns
    -------
    glfParams : namedtuple
        named tuple containing the Gaussian Lens Formula parameters

    Notes
    ----- 
    1. Both object and image distances are considered positive.  
    2. Assumes frontoparallel configuration 

    Examples
    --------
    >>> gaussian_lens_formula(u=30, v=None, f=10)
    glfParams(u=30, v=15.0, f=10)
    >>> gaussian_lens_formula(u=30, v=15)
    glfParams(u=30, v=15, f=10.0)
    >>> gaussian_lens_formula(u=1e20, f=10)
    glfParams(u=1e+20, v=10.0, f=10)
    """
    glfParams = co.namedtuple('glfParams', ['u', 'v', 'f'])
    def unknown_distance(knownDistance, f):
        try: 
            unknownDistance = (knownDistance * f)/(knownDistance - f)
        except ZeroDivisionError:
            unknownDistance = infinity 
        return unknownDistance

    def unknown_f(u, v):
        return (u*v)/(u+v)

    if sum(i is None for i in [u, v, f]) > 1:
        raise ValueError('At most only one parameter can be None')

    if f is None:
        if not u or not v:
            raise ValueError('f cannot be determined from input')
        else:
            f = unknown_f(u, v)
    else:
        if u is None:
            u = unknown_distance(v, f)
        else:
            v = unknown_distance(u, f)
    return glfParams(u, v, f)


def pupil_centric_lens_formula(u=None, v=None, f=None, mp=1, infinity=10e20):
    """return the third value of the pupil centric lens formula, given any two

    Parameters
    ----------
    u : float, optional
        object distance from entrance pupil position 
    v : float, optional
        image (geometrical focus) distance from exit pupil position
    f : float, optional
        focal length
    mp : float, optional 
        pupil magnification
    infinity : float
        numerical value to represent infinity (default=10e20)

    Returns
    -------
    pclfParams : namedtuple
        named tuple containing the Pupil Centric Lens Formula parameters

    Notes
    ----- 
    1. Both object and image distances are considered positive.
    2. Assumes frontoparallel configuration
    3. The pupil magnification (mp) is the ratio of the paraxial exit pupil size 
       to the paraxial entrance pupil size    

    Examples
    --------
    >>> pupil_centric_lens_formula(u=1016.0, v=None, f=24)
    pclfParams(u=1016.0, v=24.580645161290324, f=24.0)
    >>> pupil_centric_lens_formula(u=1016.0, v=24.58064516129, f=None)
    pclfParams(u=1016.0, v=24.58064516129, f=23.999999999999694)
    >>> pupil_centric_lens_formula(u=504.0, v=None, f=24.0, mp=2.0)
    pclfParams(u=504.0, v=49.170731707317074, f=24.0)
    >>> pupil_centric_lens_formula(u=535.6319, v=None, f=24.0, mp=0.55)
    pclfParams(u=535.6319, v=14.370742328796812, f=24.0)
    """
    pclfParams = co.namedtuple('pclfParams', ['u', 'v', 'f'])

    if sum(i is None for i in [u, v, f]) > 1:
        raise ValueError('At most only one parameter out of u, v & f can be None')

    if f is None:
        if not u or not v:
            raise ValueError('f cannot be determined from input')
        else:
            f = (mp*u*v)/(mp**2 * u + v)
    else:
        if u is None:
            try:
                u = (1.0/mp)*(v*f)/(v - mp*f)
            except ZeroDivisionError:
                u = infinity
        else:
            try:
                v = (mp**2 * f * u)/(mp * u - f)
            except ZeroDivisionError:
                v = infinity
    return pclfParams(u, v, f)


#%% Helper functions to draw the cardinal and pupil planes
def set_surface_semidia(ln, surf, value=0):
    """set the surface `surf` semi-diameter to value. 

    This requires a special function because we need to first set 
    the solve on the semi-diameter to 'fixed', else Zemax will automatically
    modify it.  

    Parameters
    ---------- 
    ln : object 
        pyzdde object 
    surf : integer
        surface number 

    Returns
    ------- 
    None
    """
    ln.zSetSolve(surf, ln.SOLVE_SPAR_SEMIDIA, ln.SOLVE_SEMIDIA_FIXED)
    ln.zSetSurfaceData(surfNum=surf, code=ln.SDAT_SEMIDIA, value=value)

def insert_dummy_surface(ln, surf, thickness=0, semidia=0, comment='dummy'):
    """helper function to insert dummy surface 

    Parameters
    ---------- 
    ln : object 
        pyzdde object 
    surf : integer
        surface number at which to insert the dummy surface 
    thickness : real, optional
        thickness of the surface 
    semidia : real, optional 
        semi diameter of the surface 
    comment : string, optional
        comment on the surface 
    """
    ln.zInsertSurface(surf)
    ln.zSetSurfaceData(surf, ln.SDAT_COMMENT, comment)
    ln.zSetSurfaceData(surf, ln.SDAT_THICK, thickness)
    set_surface_semidia(ln, surf, semidia)


def draw_plane(ln, space='img', dist=0, surfName=None, semiDia=None):
    """function to draw planes at the points specified by "dist"
    
    Parameters
    ----------
    ln : pyzdde object
        active link object
    space : string (`img` or `obj`), optional
        image space or object space in which the plane is specified. 'img' for 
        image space, 'obj' for object space. This info is required because 
        Zemax returns distances that are measured w.r.t. surface 1 (@LDE) in 
        object space, and w.r.t. IMG in image space. See the Assumptions.
    dist : float, optional
        distance along the optical axis of the plane from surface 2 (@LDE) if 
        `space` is `obj` else from the IMG surface. This assumes that surface 1
        is a dummy surface
    surfName : string, optional
        name to identify the surf in the LDE, added to the comments column
    semiDia : real, optional
        semi-diameter of the surface to set 
        
    Returns
    -------
    None
    
    Assumptions (important to read)
    -------------------------------
    The function assumes (for the purpose of this study) that surface 1 @ LDE is 
    a dummy surface at certain distance preceding the first actual lens surface. 
    This enables the rays entering the lens to be visible in the Zemax layout 
    plots even if the object is at infinity. So the function inserts the planes 
    (and their associated dummy surfaces) beginning at surface 2.
    """
    numSurf = ln.zGetNumSurf()
    inSurfPos = numSurf if space=='img' else 2 # assuming that the first surface will be a dummy surface
    insert_dummy_surface(ln, inSurfPos, dist, 0, 'dummy')
    ln.zInsertSurface(inSurfPos+1)
    ln.zSetSurfaceData(inSurfPos+1, ln.SDAT_COMMENT, surfName)
    if semiDia:
        set_surface_semidia(ln, inSurfPos+1, semiDia)
    thickSolve, pickupSolve = 1, 5
    frmSurf, scale, offset, col = inSurfPos, -1, 0, 0
    ln.zSetSolve(inSurfPos+1, thickSolve, pickupSolve, frmSurf, scale, offset, col)

def get_cardinal_points(ln):
    """Returns the distances of the cardinal points (along the optical axis)
    
    Parameters
    ----------
    ln : object
        PyZDDE object
    
    Returns
    -------
    fpObj : float
        distance of object side focal point from surface 1 in the LDE, 
        irrespective of which surface is defined as the global reference  
    fpImg : float
        distance of image side focal point from IMG surface
    ppObj : float
        distance of the object side principal plane from surface 1 in the 
        LDE, irrespective of which surface is defined as the global 
        reference surface 
    ppImg : float
        distance of the image side principal plane from IMG
    
    Notes
    -----
    1. The data is consistant with the cardinal data in the Prescription file
       in which, the object side data is with respect to the first surface in the LDE. 
    2. If there are more than one wavelength, then the distances are averaged.  
    """
    zmxdir = os.path.split(ln.zGetFile())[0]
    textFileName = os.path.join(zmxdir, "tmp.txt") 
    sysProp = ln.zGetSystem()
    numSurf = sysProp.numSurf
    ln.zGetTextFile(textFileName, 'Pre', "None", 0)
    line_list = pyz._readLinesFromFile(pyz._openFile(textFileName))
    ppObj, ppImg, fpObj, fpImg = 0.0, 0.0, 0.0, 0.0
    count = 0
    for line_num, line in enumerate(line_list):
        # Extract the Focal plane distances
        if "Focal Planes" in line:
            fpObj += float(line.split()[3])
            fpImg += float(line.split()[4])
        # Extract the Principal plane distances.
        if "Principal Planes" in line and "Anti" not in line:
            ppObj += float(line.split()[3])
            ppImg += float(line.split()[4])
            count +=1  #Increment (wavelength) counter for averaging
    # Calculate the average (for all wavelengths) of the principal plane distances
    # This is only there for extracting a single point ... ideally the design
    # should have just one wavelength define!
    if count > 0:
        fpObj = fpObj/count
        fpImg = fpImg/count
        ppObj = ppObj/count
        ppImg = ppImg/count
    # # Delete the temporary file
    pyz._deleteFile(textFileName)
    cardinals = co.namedtuple('cardinals', ['Fo', 'Fi', 'Ho', 'Hi'])
    return cardinals(fpObj, fpImg, ppObj, ppImg)
      
def draw_pupil_cardinal_planes(ln, firstDummySurfOff=40, cardinalSemiDia=1.2, push=True, printInfo=True):
    """Insert paraxial pupil and cardinal planes surfaces in the LDE for rendering in
    layout plots.
    
    Parameters
    ----------
    ln : object
        pyzdde object
    firstDummySurfOff : float, optional 
        the thickness of the first dummy surface. This first dummy surface is 
        inserted by this function. See Notes.
    cardinalSemiDia : float, optional 
        semidiameter of the cardinal surfaces. (Default=1.2) 
    push : bool, optional
        push lens in the DDE server to the LDE
    printInfo : bool, optional
        if True (default), the function also prints information about the 
        locations of the cardinal planes and pupils
        
    Assumptions
    -----------
    The function assumes that the lens is already focused appropriately,
    for either finite or infinite conjugate imaging. 
    
    Notes
    -----
    1. 'first dummy surface' is a dummy surface in LDE position 1 (between the 
        OBJ and the actual first lens surface) whose function is show the input 
        rays to the left of the first optical surface.
    2. The cardinal and pupil planes are drawn using standard surfaces in the LDE. 
       To ensure that the ray-tracing engine does not treat these surfaces as real 
       surfaces, we need to instruct Zemax to "ignore" rays to these surfaces. 
       Unfortunately, we cannot do it programmatically. So, after the planes have 
       been drawn, we need to manually do the following:
           1. 2D Layout settings
               a. Set number of rays to 1 or as needed
           2. For the pupil (ENPP and EXPP) and cardinal surfaces (H, H', F, F'), 
              and the dummy surfaces (except for the dummy surface named "dummy 2 
              c rays" go to "Surface Properties" >> Draw tab
               a. Select "Skip rays to this surface" 
           3. Set field points to be symmetric about the optical axis
    3. For clarity, the semi-diameters of the dummy sufaces are set to zero.
    """
    ln.zSetWave(0, 1, 1)
    ln.zSetWave(1, 0.55, 1)
    # insert dummy surface at 1 for showing the input ray
    ln.zRemoveVariables()
    # before inserting surface check to see if the object is at finite 
    # distance. If the object is at finite distance, inserting a dummy 
    # surface with finite thickness will change the image plane distance.
    # so first decrease the thickness of the object surface by the 
    # thickness of the dummy surface
    objDist = ln.zGetSurfaceData(surfNum=0, code=ln.SDAT_THICK)
    assert firstDummySurfOff < objDist, ("dummy surf. thick ({}) must be < "
                                         "than obj dist ({})!".format(firstDummySurfOff, objDist))
    if objDist < 1.0E+10:
        ln.zSetSurfaceData(surfNum=0, code=ln.SDAT_THICK, value=objDist - firstDummySurfOff)
    
    insert_dummy_surface(ln, surf=1, thickness=firstDummySurfOff, semidia=0, comment='dummy 2 c rays')
    ln.zGetUpdate()
    # Draw Exit and Entrance pupil planes
    expp = ln.zGetPupil().EXPP
    draw_plane(ln, 'img', expp, "EXPP")
    enpp = ln.zGetPupil().ENPP
    draw_plane(ln, 'obj', enpp - firstDummySurfOff, "ENPP")

    # Get and draw the Principal planes
    fpObj, fpImg, ppObj, ppImg = get_cardinal_points(ln)    
    draw_plane(ln,'img', fpImg, "F'", cardinalSemiDia)
    draw_plane(ln,'obj', fpObj - firstDummySurfOff, "F", cardinalSemiDia)
    draw_plane(ln,'img', ppImg, "H'", cardinalSemiDia)
    draw_plane(ln,'obj', ppObj - firstDummySurfOff, "H", cardinalSemiDia)

    # Check the validity of the distances
    ppObjToEnpp = ppObj - enpp
    ppImgToExpp = ppImg - expp
    focal = ln.zGetFirst().EFL
    v = gaussian_lens_formula(u=ppObjToEnpp, v=None, f=focal).v
    ppObjTofpObj = ppObj - fpObj
    ppImgTofpImg = ppImg - fpImg
    
    if printInfo:
        print("Textual information about the planes:\n")
        print("Exit pupil distance from IMG:", expp)
        print("Entrance pupil from Surf 1 @ LDE:", enpp)
        print("Focal plane obj F from surf 1 @ LDE: ", fpObj, "\nFocal plane img F' from IMA: ", fpImg)
        print("Principal plane obj H from surf 1 @ LDE: ", ppObj, "\nPrincipal plane img H' from IMA: ", ppImg)
        print("Focal length: ", focal)
        print("Principal plane H to ENPP: ", ppObjToEnpp)
        print("Principal plane H' to EXPP: ", ppImgToExpp)
        print("Principal plane H' to EXPP (abs.) calc. using lens equ.: ", abs(v))
        print("Principal plane H' to rear focal plane: ", ppObjTofpObj)
        print("Principal plane H to front focal plane: ", ppImgTofpImg)
        print(("""\nCheck "Skip rays to this surface" under "Draw Tab" of the """
               """surface property for the dummy and cardinal plane surfaces. """
               """See Docstring Notes for details."""))
    if push:
        ln.zPushLens(1)

def insert_cbs_to_tilt_lens(ln, lastSurf, firstSurf=2, pivot='ENPP', offset=0, push=True):
    """function to insert appropriate coordinate break and dummy surfaces 
    in the LDE for tilting the lens about a pivot. 
    
    Parameters
    ----------
    ln : object
        pyzdde object
    lastSurf : integer
        the last surface that the tilt coordinate-break should include. This 
        surface is normally the image side principal plane surface H' 
    firstSurf : integer, optional, default=2
        the first surface which the tilt coordinate-break should include. 
        Generally, this surface is 2 (the dummy surface preceding the 
        object side principal plane H)    
    pivot : string, optional 
        indicate the surface about which to rotate the lens group. Currently only 
        ENPP (with an offset from the ENPP) has been implemented
    offset : real, optional
        offset (in lens units) to offset the actual pivot point from the `pivot`
    push : bool
        push lens in the DDE server to the LDE

    Returns
    ------- 
    cbs : 2-tuple of integers 
        the first and the second coordinate breaks. Also, if `push` is True, 
        then the LDE will be updated  

    Notes
    ----- 
    1. A dummy surface will be inserted at Surface 2 to move the CB to the 
       pivot point.
    2. Check "skip rays to this surface" for the first and last dummy surfaces 
       inserted by this function.
    3. The layout will display all the surfaces. This function is needed in spite 
       of the `ln.zTiltDecenterElements()` function because it is designed to tilt
       all the the cardinal and associated dummy surfaces in between that may appear
       before or after the actual lens surfaces in the LDE. Also, the pivot point is 
       generally not about the lens surface.
    4. Use this function to rotate the lens group about a pivot point (PIVOT). To 
       rotate the image plane use a CB infront of the IMA surface. 

    Assumptions (weak)
    -----------------
    The following assumptions need not be strictly followed.
    1. Surface 0 is the object surface which may or may not be at infinity 
    2. Surface 1 is a dummy surface for seeing ray visibility.  
    """
    ln.zRemoveVariables()
    gRefSurf = ln.zGetSystem().globalRefSurf
    assert ln.zGetSurfaceData(gRefSurf, ln.SDAT_THICK) < 10e10, 'Change global ref' 

    if pivot=='ENPP':
        # estimate the distance from the firstSurf to ENPP
        enpp = ln.zGetPupil().ENPP # distance of entrance pupil from surface 1
        enppInGRef = ln.zOperandValue('GLCZ', 1) + enpp
        firstSurfInGRef = ln.zOperandValue('GLCZ', firstSurf)
        firstSurfToPivot = enppInGRef - firstSurfInGRef + offset
        cbFirstSurf = firstSurf + 1
        # insert dummy surface to move to the pivot position where the CB will be applied 
        cmtstr = 'Move to pivot (offset from ENPP)' if offset else 'Move to ENPP'
        insert_dummy_surface(ln, surf=firstSurf, thickness=firstSurfToPivot, 
                             semidia=0, comment=cmtstr)
        lastSurf +=1
        # insert dummy surface to show pivot ... 
        insert_dummy_surface(ln, surf=firstSurf+1, thickness=0, semidia=1.0, comment='PIVOT')
        cbFirstSurf +=1
        lastSurf +=1
        # insert coordinate breaks
        cb1, cb2, _ = ln.zTiltDecenterElements(firstSurf=cbFirstSurf, lastSurf=lastSurf, 
                                               cbComment1='Lens tilt CB',
                                               cbComment2='Lens restore CB', 
                                               dummySemiDiaToZero=True)
        # set solve on cb1 surface to move back to firstSurf
        ln.zSetSolve(cb1, ln.SOLVE_SPAR_THICK, ln.SOLVE_THICK_PICKUP, 
                     firstSurf, -1.0, 0.0, 0)

    else:
        raise NotImplementedError("Option not Implemented.")

    if push:
        ln.zPushLens(1)

    return (cb1, cb2)


def show_grid_distortion(n=11):
    """Plots intersection points of an array of real chief rays and paraxial
    chief rays from the object field with the image surface. 
    
    Rays are traced on the lens in the LDE (and not in the DDE server). The 
    paraxial chief ray intersection points forms the reference field. Comparing 
    the two fields gives a sense of the geometric distortion produced by the lens.  
    
    Parameters
    ----------
    n : integer, optional
        number of field points along each x- and y-axis. Total number
        of rays traced equals n**2
    
    Returns
    -------
    None
        A matplotlib scatter plot displays the intersection points, against
        the paraxial intersection points of the chief ray on that specified 
        surface. Therefore, the plot shows the distortion in the system. 

    Notes
    -----
    1. The reference field in Zemax's Grid Distortion plot is generated by 
       tracing a real chief ray very close to the optical axis, and then 
       scaling appropriately. 
    2. TODO: 
       1. Show rays that are vignetted.    
    """
    xPara, yPara, _, _, _     = get_chief_ray_intersects(n, False)
    xReal, yReal, _, err, vig = get_chief_ray_intersects(n, True)
    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6,6), dpi=120)
    ax.scatter(xPara, yPara, marker='o', s=10, facecolors='none', edgecolors='b', 
               alpha=0.8, zorder=12)
    ax.scatter(xReal, yReal, marker='x', s=40, c='r', alpha=0.9, zorder=15)
    # these two lines is dependent on the way the hx, hy grid is created in the 
    # function get_chief_ray_intersects()
    xGridPts = xPara[:n]
    yGridPts = yPara[::n] 
    ax.vlines(xGridPts, ymin=min(yGridPts), ymax=max(yGridPts), 
              zorder=2, colors='#CFCFCF', lw=0.8)
    ax.hlines(yGridPts, xmin=min(xGridPts), xmax=max(xGridPts), 
              zorder=2, colors='#CFCFCF', lw=0.8)
    ax.set_aspect('equal')
    ax.axis('tight')
    plt.show()


def get_chief_ray_intersects(n=11, real=True, surf=-1):
    """Returns the points of intersection of the chief rays from the 
    object field (normalized) with the specified surface. 

    Parameters
    ----------
    n : integer, optional
        number of field points along each x- and y-axis. Total number
        of rays traced equals n**2
    real : bool, optional 
        If `True` (default) then real chief rays are traced, 
        If `False`, then paraxial chief rays are traced 
    surf : integer, optional
        surface number. -1 (default) indicates image plane
    
    Returns
    ------- 
    x : ndarray float32
        x-intersects of the chief rays with `surf`
    y : ndarray of float32
        y-intersects of the chief rays with `surf`
    z : ndarray of float32
        z-incersects of the chief rays with `surf`
    err : ndarray of int16
        error code
    vig : ndarray of uint8
        vignetting code 

    Notes
    ----- 
    TODO: 
    1. How to handle errors that occurs during the ray tracing.
    2. The function seems to work only if the field normalization is set 
       to "radial". Need to fix it for rectangular field normalization. 
    """
    # create uniform square grid in normalized field coordinates
    nx = np.linspace(-1, 1, n)
    hx, hy = np.meshgrid(nx, nx)
    hx = hx.flatten().tolist()
    hy = hy.flatten().tolist()
    
    # trace the ray
    mode = 0 if real else 1
    rayData = at.zGetTraceArray(numRays=n**2, hx=hx, hy=hy, mode=mode, surf=surf)
    # parse ray traced data
    err = np.array(rayData[0], dtype=np.int16)
    vig = np.array(rayData[1], dtype=np.uint8)
    x = np.array(rayData[2], dtype=np.float32)
    y = np.array(rayData[3], dtype=np.float32)
    z = np.array(rayData[4], dtype=np.float32)
    return x, y, z, err, vig 

def plot_chiefray_intersects(ln, cb, tiltXY, pushNewLens=True):
    """plot chief-ray intersects for various rotations of the lens about a pivot point.
    
    Parameters
    ----------
    ln : object
        pyzdde object
    cb : integer 
        Element tilt coordinate break (the first CB) surface number 
    tiltXY : list of 2-tuples
        each element of the list is a tuple containing the tilt-about-x 
        angle, tilt-about-y angle in degrees of the lens about a 
        pivot point.
        e.g. tiltXY = [(0,0), (10, 0), (0, 10)]
        Max length = 6
    pushNewLens : boolean, optional 
        see notes 
        
    Returns
    -------
    None : matplotlib figure

    Notes
    -----
    1. This function uses the array ray tracing module functions for generating 
       the chief-ray--image-plane intersection points. To do so, it needs to push 
       the lens in the DDE server into the LDE. By default, following the ray 
       tracing, a new lens is loaded into the LDE. Please make sure that there 
       are no unsaved/work-in-progress lens files in the LDE before invoking this 
       function.
    2. This function always traces "real" rays. This is because paraxial ray 
       tracing doesn't seem to be affected by lens tilts, i.e. the ray-intersect 
       pattern is constant for all rotations of the lens.  
    """
    tiltAbtXParaNum = 3
    tiltAbtYParaNum = 4
    cols = ['b', 'r', 'g', 'm', 'c', 'y']
    mkrs = ['o', 'x', '+', '1', '2', '3']
    n = 11
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
    for i, angle in enumerate(tiltXY):
        ln.zSetSurfaceParameter(surfNum=cb, param=tiltAbtXParaNum, value=angle[0])
        ln.zSetSurfaceParameter(surfNum=cb, param=tiltAbtYParaNum, value=angle[1])
        ln.zGetUpdate()
        # push lens into the LDE as array tracing occurs in the LDE
        ln.zPushLens(1)
        x, y, _, err, vig = get_chief_ray_intersects(n=n, real=True)
        legTxt = r'$\alpha_x, \alpha_y = {}^o,{}^o$'.format(angle[0], angle[1])
        if i:
            ax.scatter(x, y, marker=mkrs[i], s=40, c=cols[i], alpha=0.9, zorder=15, 
                       label=legTxt)
        else:
            ax.scatter(x, y, marker=mkrs[i], s=10, facecolors='none', 
                       edgecolors=cols[i], alpha=0.8, zorder=12, label=legTxt)
            xGridPts = x[:n]
            yGridPts = y[::n] 
            ax.vlines(xGridPts, ymin=min(yGridPts), ymax=max(yGridPts), 
                      zorder=2, colors='#CFCFCF', lw=0.8)
            ax.hlines(yGridPts, xmin=min(xGridPts), xmax=max(xGridPts), 
                      zorder=2, colors='#CFCFCF', lw=0.8)
    # finally load a new lens into the LDE
    if pushNewLens:
        ln.zNewLens()
        ln.zPushLens(1)
    ax.set_aspect('equal')
    ax.axis('tight')
    ax.set_ylabel(r'$\bf{\acute{y}}\,\it{(mm)}$', fontsize=15)
    ax.set_xlabel(r'$\bf{\acute{x}}\,\it{(mm)}$', fontsize=15)
    ax.legend(fontsize=14, scatterpoints=1, markerscale=1., scatteryoffsets=[0.5], mode='expand',
              ncol=len(tiltXY), loc='upper left', bbox_to_anchor=(0.09, 0.965, 0.84, 0.005), 
              bbox_transform=fig.transFigure, handletextpad=0.5, handlelength=0.9)
    plt.show()
    



#%% Functions for focal / angular sweep to EDOF simulation using Zemax

# Meta-data schema for storing frames in hdf5 file
# ------------------------------------------------
# / (root Group)
#     Attribute: zemax file (string)
#     Attribute: obj_sharp_focus_distance (real)
#     Attribute: lens_tilt_about_x
#     Attribute: lens_tilt_about_y
#     Attribute: img_sharp_focus_distance (real)
#     Attribute: img_tilt_about_x
#     Attribute: img_tilt_about_y
#     Attribute: img_sim_field_height
#     Attribute: img_sim_oversampling
#     Attribute: img_sim_wavelength
#     Attribute: img_sim_pupil_sampling
#     Attribute: img_sim_image_sampling
#     Attribute: img_sim_psfx_points
#     Attribute: img_sim_psfy_points
#     Attribute: img_polarizatoin
#     Attribute: img_sim_aberrations
#     Attribute: img_sim_relative_illumination
#     Attribute: img_sim_fixed_apertures
#     Attribute: img_sim_reference
#     Attribute: img_sim_pixel_size
#     Attribute: img_sim_xpixels
#     Attribute: img_sim_ypixels


# attributes for each frame
# 1. defocus_waves (only makes sense for fronto-parallel imaging)
# 2. 

def get_image_plane_shifts(nearObj=800, midObj=1000, farObj=1200, fl=24, num=10):
    """Returns the image plane delta shifts in both forward and 
    backward directions from the base position of the image plane.
    
    The "base position" is the position of the image plane for which 
    the middle object is in geometrical focus.
    
    Parameters
    ----------
    farObj : real
        distance along the optical axis for furthest object
    midObj : real
        distance along the optical axis for the middle object
    nearObj : real
        distance along the optical axis for the nearest object
    fl : real
        effective focal length 
    num : integer
        number of images in focal stack
        
    Returns
    -------
    imgDelta : list
        sorted list containing the delta image plane shifts with a 
        guaranteed 0 delta shift value that corresponds to the 
        "base position" of the image plane
        
    Notes
    -----
    1. The inputs distances are in the object side. The output
       image plane shifts are in the image side.
    2. It is highly improbable, but possible, that the length of the
       returned list is one less than `num`. This unlikely event 
       occurs if `num` is even and distance between the far and 
       near object focus in the image side can be evenly divided.
    """
    midFocalImg = gaussian_lens_formula(u=midObj, f=fl).v
    nearFocalImg = gaussian_lens_formula(u=nearObj, f=fl).v
    farFocalImg = gaussian_lens_formula(u=farObj, f=fl).v
    a = farFocalImg - midFocalImg
    b = nearFocalImg - midFocalImg
    imgDelta = np.linspace(a, b, num - 1).tolist()
    if all(imgDelta):
        imgDelta = imgDelta + [0,]
        imgDelta.sort()
    return imgDelta

# Here is a quick and dirty helper function to estimate the size of the PSF 
# in the image plane using Zemax

def get_spot_dia(ln,  spotType='rms', whichWave='pri', hy=0, field=1, frac=0.98):
    """returns diffraction limited, RMS spot or encircled energy fraction based
    diameter in the image plane in microns.

    Parameters
    ----------
    ln : object 
        pyzdde link object
    spotType : 3 letter string code ('rms', 'diff', 'enc')
        'diff' for diffraction limited airy spot diameter, 
        'rms' for RMS spot diameter
        'enc' : greater of diffraction encircle (ensquared) energy or 
        and geometric encircle energy  
    whichWave : 3 letter string code ('pri' or 'gre')
        wavelength - 'pri' for primary wave, 'gre' for greatest.
        (applicable only for diffraction limited spot computation. `rms` 
         and `enc` uses weighted polychromatic wavelength)
    hy : real (0 <= hy <= 1), optional 
        normalized field height (applicable only for `rms` spot). 
    field : integer, optional  
        field number (applicable only for `enc` computation). 
    frac : real (0 < frac <=1), optional 
        fraction for encircled energy calculation  
        
    Returns
    -------
    spotDia : real
        spot diameter in microns

    Notes
    ----- 
    1. The diffraction stop ('diff') returns on-axis airy diameter 
    2. For paraxial systems, the 'rms' spot radius will zero or very close
       to zero. Use diffraction limited spot diameter or encircled energy 
       for such systems.
    3.  
    """
    if spotType =='diff':    
        if whichWave=='pri':
            wave = ln.zGetWaveTuple().wavelengths[ln.zGetPrimaryWave()-1]
        else:
            waveTuple = list(ln.zGetWaveTuple().wavelengths)
            waveTuple.sort()
            wave = waveTuple[-1]
        fnum = ln.zGetFirst().paraWorkFNum
        diffLimSpotDia = 2.44*wave*fnum # in microns
        return diffLimSpotDia
    elif spotType=='rms': # RMS spot 
        rmsOperC = ['RSCE', 'RSCH']  # GQ usg. circ. grid
        rmsOperR = ['RSRE', 'RSRH']  # usg. rect. grid of rays
        rmsSpotC, rmsSpotR = [], []
        hx = 0
        hy = hy if hy is not None else 0
        ring, samp, wave, hx = 6, 10, 0, 0 
        for oper in rmsOperC:
            rmsSpotC.append(ln.zOperandValue(oper, ring, wave, hx, hy))
        for oper in rmsOperR:
            rmsSpotR.append(ln.zOperandValue(oper, samp, wave, hx, hy))
        rmsSpots = rmsSpotC + rmsSpotR
        rmsSpotDia = 2.0*np.median(rmsSpots)*1000 # microns
        return rmsSpotDia
    elif spotType=='enc': # encircled energy 
        samp, wave, etype = 3, 0, 4
        field = field if field is not None else 1
        denc = 2.0*ln.zOperandValue('DENC', samp, wave, field, frac, etype, 3, 0, 0)
        genc = 2.0*ln.zOperandValue('GENC', samp, wave, field, frac, etype, 0, 0, 0)
        return genc if genc > denc else denc 
    else:
        raise ValueError, 'Incorrect spot type'

# the minimum (diffraction limited spot at best focus) should determine the field height 
# based on a criterion of number of pixels per spot. The maximum spot size should determine
# the image sampling

def get_min_max_PSF_size(ln, imgDelta, frac=0.98):
    """returns the minimum and maximum spot size (PSF) in the 
    object space from the set of spot sizes computed over the
    given set of image plane shifts
    
    The spot size in the object space is obtained by scaling the 
    image space spot size by the transverse magnification
    
    Parameters
    ----------
    ln : object
        pyzdde object
    imgDelta : list 
        list of image plane shifts (deltas) from the baseline 
        image plane configuration
    frac : real (0 < frac <= 1)
        fraction of ensquared energy 
        
    Returns
    -------
    spots : tuple
        a tuple consisting of the minimum and maximum spot 
        sizes in microns
    
    Assumptions
    -----------
    The optical system is focal and in finite imaging configuration
    
    Notes
    -----
    1. It is generally sufficient to inspect the psf spot sizes only 
       for the extreme shifts on either sides of the best focus and
       the best focus. i.e restrict the length of `imgDelta` to just 
       three configurations.
    2. Currently the computation is performed only for on-axis field
       point. However, it can be changed easily by providing another
       input parameter to this function and then calling 
       `gm.get_spot_dia()` accordingly.
    """
    # insert surface to impart image delta shifts of the image plane
    imgDeltaSurf = totSurf = ln.zGetNumSurf()
    ln.zInsertSurface(surfNum=totSurf)
    ln.zSetSemiDiameter(surfNum=imgDeltaSurf, value=0)
    spot = []  
    for delta in imgDelta:
        #print('image delta:', delta)
        ln.zSetThickness(surfNum=imgDeltaSurf, value=delta)
        ln.zGetUpdate()
        ensqSpotImg = get_spot_dia(ln, spotType='enc', frac=frac)
        #print('ensqSpotImg', ensqSpotImg)
        ensqSpotObj = ensqSpotImg /abs(ln.zGetMagnification())
        spot.append(ensqSpotObj)
    # clean up the inserted surface
    ln.zDeleteSurface(surfNum=imgDeltaSurf)
    spot = np.array(spot)
    return spot.min(), spot.max()

def obj_field_height(ppp, ypix, minSpotDia):
    """determines the maximum field height for image simulation
    in zemax based on a minimum pixels per PSF criterion
    
    Parameters
    ----------
    ppp : integer 
        pixels per PSF (a design choice / criterion)
    ypix : integer 
        number of pixels along the height of the source bitmap
    minSpotDia : real 
        diameter of the minimum spot size at best focus
    
    Returns
    -------
    h : real 
        height (in units of `minSpotDia`) of field
    
    Logic
    -----
    
      h                     minSpotDia
    ----- = objPixelSize = ------------
     ypix                     ppp
     
    """
    return ypix*minSpotDia/ppp

def image_sampling(h, ypix, maxSpotDia):
    """determine the image sampling to ensure sufficient 
    samples for worst case PSF
    
    Parameters
    ----------
    h : real 
        field height in millimeters
    ypix : integer 
        number of pixels along the height of the source bitmap
    maxSpotDia : real 
        diameter of the worst case PSF in millimeters
    
    Returns
    -------
    numPixels : integer 
        minimum number of pixels to select for image sampling
    
    """
    pixelHeight = h/ypix
    numPixels = int(maxSpotDia/pixelHeight)
    return numPixels

def get_detector_settings(h, xpix, ypix, fl, xfield, umid, unear=None, ufar=None):
    """computes and returns appropriate detector settings to be used for the 
    Zemax Image Simulation tool.  
    
    Parameters
    ----------
    h : real 
        field height of the source bitmap image in image 
        simulation setting, in mm
    xpix : integer
        number of pixels along the width in source bitmap
    ypix : integer 
        number of pixels along the height in source bitmap
    f : real
        focal length of the lens
    xfield : real
        x field (as object height) in fields settings
    umid : real
        distance along the optical axis for the middle 
        object plane (that is in focus)
    unear : real, optional
        distance along the optical axis for the nearest 
        object plane. If `None`, then `unear` is assumed 
        to be equal to `umid` for detector size computation. 
    ufar : real, optional 
        distance along the optical axis for the farthest 
        object plane. If `None`, then `ufar` is assumed to 
        be equal to `umid` for detector pixel size computation.
        
    Returns
    -------
    detPixelSize : real 
        pixel size of detector in millimeters
    detXpixels : integer
        X Pixels of detector
    detYpixels : integer
        Y Pixels of detector
    
    Note
    ----
    1. Zemax (and so do we) assumes square pixels
    2. The function assumes that the field setting contains 
       ± xfield 
    """
    unear = unear if unear is not None else umid
    ufar = ufar if ufar is not None else umid 
    aratio = xpix/ypix
    w = aratio*h   # width in obj space
    pixHeightObj = h/ypix
    # calculation on the middle plane (that is in focus)
    tmag = fl/(umid - fl)
    imgDist = gaussian_lens_formula(u=umid, v=None, f=fl).v
    # calcuate the detector pixel size
    #detPixelSize = pixHeightObj*abs(tmag)  
    # the smallest detector size required to prevent image simulation
    # artifacts should be determined from the smallest transverse magnification,
    # which is the magnification associated with the largest object plane 
    # distance.
    tmagFar = imgDist/ufar   # note that the far object is not in focus
    detPixelSize = pixHeightObj*tmagFar 

    # calculate detector width and height for the largest
    # magnification
    #tmagNear = fl/(unear -fl)
    tmagNear = imgDist/unear   # note that the near object is not in focus
    imgHeight = h*abs(tmagNear) 
    imgWidth = w*abs(tmagNear)
    detXPixels = int((2.0*xfield*abs(tmagNear) + imgWidth)/detPixelSize) + 30  
    detYPixels = int(imgHeight/detPixelSize) + 60 # 25 pixels of guard top & bottom
    return detPixelSize, detXPixels, detYPixels

def grid_of_square_dots(pixx=640, pixy=480, numx=7, numy=7, size=1, ch=3,
                        interSpread='max'):
    """returns a grid-of-square-dots array as Numpy's ndarray 
    
    Parameters
    ----------
    pixx : integer 
        number of pixels along x (horizontal or number of columns)
    pixy : integer 
        number of pixels along y (vertical or number of row)
    numx : integer 
        number of square dots along x
    numy : integer 
        number of square dots along y
    size : integer (odd)
        number of pixels on the square sides. The square is 
        always symmetric about zero. 
    ch : integer 
        number of color planes. Usually 1 or 3
    interSpread : string ('max', 'uni')
        if 'uni' intermittent space between the dots and the edges
        along each axis are equal (may be off by a pixel)
        if 'max' the intermittent space between the dots are 
        maximized at the cost of the space along the boundary (i.e
        the guard band is reduced)
        
    """
    if interSpread == 'max': 
        uniDeltaXby2 = np.floor(0.5*pixx/(numx-1)) if numx > 1 else np.floor(0.5*pixx)
        uniDeltaYby2 = np.floor(0.5*pixy/(numy-1)) if numy > 1 else np.floor(0.5*pixy)
        anyPerPixX = np.floor(0.15*pixx) if numx > 1 else np.floor(0.5*pixx)
        anyPerPixY = np.floor(0.15*pixy) if numy > 1 else np.floor(0.5*pixy)
        guardX = anyPerPixX if anyPerPixX < uniDeltaXby2 else uniDeltaXby2
        guardY = anyPerPixY if anyPerPixY < uniDeltaYby2 else uniDeltaYby2       
        xg = np.floor(np.linspace(guardX,  pixx - guardX, numx)).astype(np.uint32)
        yg = np.floor(np.linspace(guardY, pixy - guardY, numy)).astype(np.uint32)
    else: # uniformly spread
        xg = np.floor(np.linspace(0, pixx, numx+2)[1:-1]).astype(np.uint32)
        yg = np.floor(np.linspace(0, pixy, numy+2)[1:-1]).astype(np.uint32)
    indicesX, indicesY = np.meshgrid(yg, xg)
    gridofdots = np.empty((pixy, pixx, ch), dtype=np.uint8)
    grid = np.zeros((pixy, pixx), dtype=np.uint8)
    for i in range(-int(size/2), int(0.9*size/2) + 1):
        for j in range(-int(size/2), int(0.9*size/2) + 1):
            grid[[indicesX-i], [indicesY-j]] = 255
    for each in range(ch):
        gridofdots[:, :, each] = grid
    return gridofdots


def simulate_depth_imaging(ln, objsurfthick, objarr, fldarr, data, cfgname, 
                           objht, over, pupsam, imgsam, psfx, psfy, pixsize, 
                           xpix, ypix, aberr=2, timeout=180, verbose=True):
    """simulate imaging with object planes at different depths. 
    
    Parameters
    ----------
    ln : object
        pyzdde object
    objsurfthick : list
        list of object surf (surfNum 0) thickness. This function changes the 
        thickness of the OBJ surface to the values specified in the `objsurfthick`
        list. The physical object-to-lens vertex distance is equal to the 
        sum of the thicknesses of the OBJ surface and surface number 1 (dummy 2 c rays) 
    objarr : list of strings 
        list of image files that represents the planar objects at the corresponding
        object distances in `objdist`. These image files must be present in the 
        folder "C:\Users\<username>\Documents\ZEMAX\IMAFiles"
    fldarr : list of integers
        list of field numbers that are used for the corresponding objects in 
        `objarr` for image simulation in Zemax. 
    data : string, ('img' or 'psf')
        if 'img' then the simulated image is returned, if 'psf', then
        the PSF grid is returned.  
    cfgname : string
        name of settings file to use
    objht : real
        object height in mm
    over : integer, [0-6] or None
        oversample value. Use 0 for none, 1 for 2X, 2 for 4x, etc.
    pupsam : integer 
        pupil sampling
    imgsam : integer
        image sampling 
    psfx : integer
        psf grid number x
    psfy: integer
        psf grid number y
    pixsize : real 
        pixel size, in mm
    xpix : integer 
        number of pixel along x
    ypix : integer
        number of pixel along y
    aberr : integer, optional 
        aberration; 0=none, 1=geometric, 2=diffraction 
    timeout : integer, optional
        timeout in ms
    verbose : bool, optional
        if True, the function prints to notify the completion of every image 
        simulation
        
    Returns
    -------
    img : ndarray
        simulated image
    mag : list 
        paraxial magnification for the objects at different object distances 
    
    Notes
    -----
    The "reference" parameter in the Image Simulation is set to "vertex".
    
    Assumptions
    -----------
    The surface 0 in the LDE is considered to be the object surface at finite distance
    
    """
    assert len(objsurfthick) == len(objarr) == len(fldarr), \
    "input arrays must be of equal lengths"
    img = np.zeros((ypix, xpix, 3), dtype='uint8')
    fpath = os.path.split(ln.zGetFile())[0]
    cfg = os.path.join(fpath, cfgname)

    showAs = 2 if data=='psf' else 0

    cfg = ln.zSetImageSimulationSettings(settingsFile=cfg, height=objht, over=over, 
                                         pupilSample=pupsam, imgSample=imgsam, 
                                         psfx=psfx, psfy=psfy, pixelSize=pixsize, 
                                         xpix=xpix, ypix=ypix, reference=1, aberr=aberr,  # reference is vertex
                                         illum=1, showAs=showAs)
    mag = []
    for thick, obj, fld in zip(objsurfthick, objarr, fldarr):
        ret = ln.zSetThickness(surfNum=0, value=thick)
        assert ret == thick
        ret = ln.zModifyImageSimulationSettings(settingsFile=cfg, image=obj, field=fld)
        imgInfo, imgData = ln.zGetImageSimulation(settingsFile=cfg, timeout=timeout)
        if data == 'img': # the PSF grid spans the field size in the object space
            assert imgInfo.xpix == xpix and imgInfo.ypix == ypix
        for i in range(3): 
            img[:, :, i] = img[:, :, i] + np.array(imgData, dtype='uint8')[:, :, i]
        mag.append(ln.zGetMagnification())
        if verbose:
            print('Time: {}. '.format(get_time()), end='')
            print(("Image sim of data type {} for obj {} for obj thick "
                   "{:2.2f} completed!".format(data, obj, thick)))
            sys.stdout.flush()
    return img, mag


def focal_stack_fronto_parallel(ln, imgDelta, objsurfthick, objarr, fldarr, objht,
                                over, pupsam, imgsam, psfx, psfy, pixsize, xpix, ypix, 
                                psfGrid=True, aberr=2, numCrImIpts=11, timeout=180, verbose=False):
    """Creates and returns a stack of simulated images for various focal distances 
    and associated metadata in HDF5 container. Also see Notes. 
    
    Parameters
    ----------
    imgDelta : list of reals
        the deltas by which the image plane is shifted from the base image plane 
        distance. 
    objsurfthick : list
        list of object surf (surfNum 0) thickness. See more in `simulate_depth_imaging()` 
    objarr : list
        list of image file names with extensions that represents the planar objects 
        at the corresponding object distances in `objdist`. See more in 
        `simulate_depth_imaging()`.  
    fldarr : list
        list of field numbers. See more in `simulate_depth_imaging()` 
    objht : real
        object height in mm
    over : integer 
        oversampling. See more in `simulate_depth_imaging()`
    pupsam : integer
        pupil sampling
    imgsam : integer
        image sampling 
    psfx : integer
        psf grid number x
    psfy: integer
        psf grid number y
    pixsize : real 
        pixel size, in mm
    xpix : integer 
        number of pixel along x
    ypix : integer
        number of pixel along y
    aberr : integer (0,1, or 2)
        aberration; 0=none, 1=geometric, 2=diffraction 
    psfGrid : bool, optional 
        if True (default), the PSF grid (in image space) is simulated and 
        embedded as a dataset. If False, the PSF grid dataset is empty.
    numCrImIpts : integer, optional (default=11) 
        number of chief-ray-IMG intersection points along each direction.
        Total number of intersection points = numCrImIpts**2
    timeout : integer, optional, default=180
        timeout in ms for each image simulation
    verbose : bool, optional
        if True, the function prints to notify the completion of every image 
        simulation

    Returns
    ------- 
    hdffileFull : string 
       the file name, including absolute path, of the hdf5file containing 
       the stack of images 

    Notes
    -----
    This functions makes use of the array ray tracing functions for getting the 
    chief-ray-image-plane ray intercept points. To do so, it needs to push the 
    lens in the DDE server into the LDE. Following the ray tracing a new lens is 
    loaded into the LDE. Please make sure that there are no unsaved lens files 
    in the LDE before calling this function.  
    """  
    TeSTCODE_LOGIC = False   # note that Zemax must be running even for True
    
    # insert a surface just before the IMA surface. Changing the thickness of 
    # this surface will change the image plane distance from the exit pupil.
    imgDeltaSurf = totSurf = ln.zGetNumSurf()
    ln.zInsertSurface(surfNum=totSurf)
    ln.zSetSemiDiameter(surfNum=imgDeltaSurf, value=0)
    
    # hdf5 file save settings
    imgstackdir = get_directory_path(['data', 'imgstack'])
    timetag = time.strftime('%Y_%m_%d_%H_%M') # year, month, day, hour, mins
    hdffile = 'fronto_para_focal_stack_{}.hdf5'.format(timetag)
    hdffileFull = os.path.join(imgstackdir, hdffile)
    samplingGrid =[32*(2**i) for i in range(10)]
    aberrtype = ('geometric' if aberr-2 else 'diffraction') if aberr else 'none'
    pmag = ln.zGetPupilMagnification()
    # create stack
    with hdf.File(hdffileFull, 'w') as f:
        globalAttribDict = {'zmxfile' : os.path.split(ln.zGetFile())[-1],
                            'focal_stack_type' : 'frontoparallel',
                            'sys_pupil_mag' : pmag,
                            'img_sim_field_height' : objht,
                            'img_sim_oversampling' : str(over),
                            'img_sim_wavelength' : 'RGB',
                            'img_sim_pupil_sampling' : samplingGrid[pupsam-1],
                            'img_sim_image_sampling' : samplingGrid[imgsam-1],
                            'img_sim_psfx_points' : psfx,
                            'img_sim_psfy_points' : psfy,
                            'img_sim_polarization' : 'None',
                            'img_sim_aberrations' : aberrtype,
                            'img_sim_relative_illumination' : 'Yes',
                            'img_sim_fixed_apertures' : 'Yes',
                            'img_sim_reference' : 'vertex',
                            'img_sim_pixel_size' : pixsize,
                            'img_sim_xpixels' : xpix,
                            'img_sim_ypixels' : ypix,
                            'cr_img_ipts_numx' : numCrImIpts,
                            'cr_img_ipts_numy' : numCrImIpts,
                           }
        set_hdf5_attribs(f, globalAttribDict)
        dataGrp = f.create_group('data')  # data group
        get_time(startClock=True)         # initiate running clock 
        for i, delta in enumerate(imgDelta):
            if verbose:
                print('Time: {}. '.format(get_time()), end='')
                print('Starting image simulation for delta = {:2.4f}'.format(delta))
                sys.stdout.flush()
            ln.zSetThickness(surfNum=imgDeltaSurf, value=delta)
            ln.zGetUpdate()
            dataSubGrp = dataGrp.create_group('{}'.format(i).zfill(3) )
            # IMAGE SIMULATION DATA
            if TeSTCODE_LOGIC:
                print('Code logic test is on.')
                img = np.random.randint(0, 255, (ypix, xpix, 3)).astype('uint8')
                mag = 1.0
            else:
                img, mag = simulate_depth_imaging(ln, objsurfthick, objarr, fldarr, 'img',
                                  'spl.cfg', objht, over, pupsam, imgsam, psfx, psfy, 
                                  pixsize, xpix, ypix, aberr, timeout, verbose)

            dataSubGrp.create_dataset('image', data=img, dtype=np.uint8)

            # PSF GRID DATA
            # Instead of asking Zemax to produce the regular PSF grid (i.e. the grid
            # of PSF in the object space), we generate a grid of dots image and 
            # run through the image simulation exactly as we did for the source image
            # to generate the "image side" PSF field 
            if psfGrid:
                god = grid_of_square_dots(pixx=xpix, pixy=ypix, numx=psfx, numy=psfy, 
                                          size=1, ch=3, interSpread='max')
                save_to_IMAdir(god, 'imgsim_god.png')
                godarr = ['imgsim_god.png']*len(objarr)

                if TeSTCODE_LOGIC:
                    print('Code logic test is on.')
                    psfgrid = grid_of_square_dots(pixx=xpix, pixy=ypix, numx=psfx, numy=psfy, 
                                                 size=10, ch=3, interSpread='max')
                else:
                    psfgrid, _ = simulate_depth_imaging(ln, objsurfthick, godarr, fldarr, 'img',
                                                    'spl.cfg', objht, over, pupsam, imgsam, psfx, 
                                                    psfy, pixsize, xpix, ypix, aberr, timeout,
                                                    verbose)
                dataSubGrp.create_dataset('psf', data=psfgrid, dtype=np.uint8)
            else:
                dataSubGrp.create_dataset('psf', data=(ypix, xpix, 3), dtype=np.uint8) # just a place holder, no actual data is stored.
            # CHIEF-RAY INTERSECT DATA 
            # push lens into the LDE as array tracing occurs in the LDE
            ln.zPushLens(1)
            crimgiptsGrp = dataSubGrp.create_group('cr_img_ipts')
            x, y, z, err, vig = get_chief_ray_intersects(n=numCrImIpts, real=True)
            crimgiptsGrp.create_dataset('x', data=x, dtype=np.float32)
            crimgiptsGrp.create_dataset('y', data=y, dtype=np.float32)
            crimgiptsGrp.create_dataset('err', data=err, dtype=np.int16)
            if verbose:
                print('Traced for chief-ray intersects...')
                sys.stdout.flush()

            # set subgroup attribute
            dataSubGrpAttribDict = {'delta_z' : delta,   # others attributes like defocus_waves, paraxial mag etc
                                    'mag' : mag,   # list of magnifications at the different depths
                                   }
            set_hdf5_attribs(dataSubGrp, dataSubGrpAttribDict)

    ln.zDeleteSurface(surfNum=imgDeltaSurf)
    return hdffileFull

def get_lens_plane_tilts(uo=1000, nearObj=800, farObj=1200, fl=24, num=10):
    """TO DO
    """
    # TO DO
    print("TO IMPLEMENT")
    return np.linspace(-8.0, 8.0, num).tolist()

def focal_stack_lens_tilts(ln, cb1, tiltX, objsurfthick, objarr, fldarr, objht, over, 
                           pupsam, imgsam, psfx, psfy, pixsize, xpix, ypix, aberr=2, 
                           psfGrid=True, numCrImIpts=11, timeout=180, verbose=False):
    """Creates and returns a stack of simulated images for various tilts of the lens  
    and associated metadata in HDF5 container. Also see Notes. 
    
    Parameters
    ----------
    cb1 : integer 
        number of the first coordinate break surface that imparts the tilt  
    tiltX : list of reals
        the deltas by which the image plane is shifted from the base image plane 
        distance. 
    objsurfthick : list
        list of object surf (surfNum 0) thickness. See more in `simulate_depth_imaging()` 
    objarr : list
        list of image file names with extensions that represents the planar objects 
        at the corresponding object distances in `objdist`. See more in 
        `simulate_depth_imaging()`.  
    fldarr : list
        list of field numbers. See more in `simulate_depth_imaging()` 
    objht : real
        object height in mm
    over : integer 
        oversampling. See more in `simulate_depth_imaging()`
    pupsam : integer
        pupil sampling
    imgsam : integer
        image sampling 
    psfx : integer
        psf grid number x
    psfy: integer
        psf grid number y
    pixsize : real 
        pixel size, in mm
    xpix : integer 
        number of pixel along x
    ypix : integer
        number of pixel along y
    aberr : integer 
        aberrations, 0=none, 1=geometric, 2=diffraction 
    psfGrid : bool, optional 
        if True (default), the PSF grid (in image space) is simulated and 
        embedded as a dataset. If False, the PSF grid dataset is empty. 
    numCrImIpts : integer, optional (default=11) 
        number of CR-IMG intersection points along each direction.
        Total number of intersection points = numCrImIpts**2
    timeout : integer, optional, default=180
        timeout in ms for each image simulation
    verbose : bool, optional
        if True, the function prints to notify the completion of every image 
        simulation

    Returns
    ------- 
    hdffileFull : string 
       the file name, including absolute path, of the hdf5file containing 
       the stack of images 

    Notes
    -----
    This functions makes use of the array ray tracing functions for getting the 
    chief-ray-image-plane ray intercept points. To do so, it needs to push the 
    lens in the DDE server into the LDE. Following the ray tracing a new lens is 
    loaded into the LDE. Please make sure that there are no unsaved lens files 
    in the LDE before calling this function.  
    """ 
    # TO DO: Remove the TeSTCODE_LOGIC logic once the code is ready 
    TeSTCODE_LOGIC = False   # Generally should be FALSE; Zemax must be running even for True
    # hdf5 file save settings
    imgstackdir = get_directory_path(['data', 'imgstack'])
    timetag = time.strftime('%Y_%m_%d_%H_%M') # year, month, day, hour, mins
    hdffile = 'lens_tilt_focal_stack_{}.hdf5'.format(timetag)
    hdffileFull = os.path.join(imgstackdir, hdffile)
    samplingGrid =[32*(2**i) for i in range(10)]
    aberrtype = ('geometric' if aberr-2 else 'diffraction') if aberr else 'none'
    pmag = ln.zGetPupilMagnification()
    # create stack
    with hdf.File(hdffileFull, 'w') as f:
        globalAttribDict = {'zmxfile' : os.path.split(ln.zGetFile())[-1],
                            'focal_stack_type' : 'lenstilts',
                            'sys_pupil_mag' : pmag,
                            'img_sim_field_height' : objht,
                            'img_sim_oversampling' : str(over),
                            'img_sim_wavelength' : 'RGB',
                            'img_sim_pupil_sampling' : samplingGrid[pupsam-1],
                            'img_sim_image_sampling' : samplingGrid[imgsam-1],
                            'img_sim_psfx_points' : psfx,
                            'img_sim_psfy_points' : psfy,
                            'img_sim_polarization' : 'None',
                            'img_sim_aberrations' : aberrtype,
                            'img_sim_relative_illumination' : 'Yes',
                            'img_sim_fixed_apertures' : 'Yes',
                            'img_sim_reference' : 'vertex',
                            'img_sim_pixel_size' : pixsize,
                            'img_sim_xpixels' : xpix,
                            'img_sim_ypixels' : ypix,
                            'cr_img_ipts_numx' : numCrImIpts,
                            'cr_img_ipts_numy' : numCrImIpts,
                           }
        set_hdf5_attribs(f, globalAttribDict)
        dataGrp = f.create_group('data')  # data group
        get_time(startClock=True)         # initiate running clock 
        for tiltCnt, tiltAbtX in enumerate(tiltX):
            if verbose:
                print('Time: {}. '.format(get_time()), end='')
                print('Starting image simulation for tiltAbtX = {:2.4f}'.format(tiltAbtX))
                sys.stdout.flush()
            ln.zSetSurfaceParameter(surfNum=cb1, param=3, value=tiltAbtX) # tilt about x
            ln.zGetUpdate()
            dataSubGrp = dataGrp.create_group('{}'.format(tiltCnt).zfill(3))
            # IMAGE SIMULATION DATA
            if TeSTCODE_LOGIC:
                print('Code logic test is on.')
                img = np.random.randint(0, 255, (ypix, xpix, 3)).astype('uint8')
                mag = 1.0
            else:
                img, mag = simulate_depth_imaging(ln, objsurfthick, objarr, fldarr, 'img',
                                                  'spl.cfg', objht, over, pupsam, imgsam, 
                                                  psfx, psfy, pixsize, xpix, ypix, aberr, 
                                                  timeout, verbose)
            dataSubGrp.create_dataset('image', data=img, dtype=np.uint8)
            # PSF GRID DATA
            # Instead of asking Zemax to produce the regular PSF grid (i.e. the grid
            # of PSF in the object space), we generate a grid of dots image and 
            # run through the image simulation exactly as we did for the source image
            # to generate the "image side" PSF field 
            if psfGrid:
                god = grid_of_square_dots(pixx=xpix, pixy=ypix, numx=psfx, numy=psfy, 
                                          size=1, ch=3, interSpread='max')
                save_to_IMAdir(god, 'imgsim_god.png')
                godarr = ['imgsim_god.png']*len(objarr)
                if TeSTCODE_LOGIC:
                    print('Code logic test is on.')
                    psfgrid = grid_of_square_dots(pixx=xpix, pixy=ypix, numx=psfx, numy=psfy, 
                                                 size=10, ch=3, interSpread='max')
                else:
                    psfgrid, _ = simulate_depth_imaging(ln, objsurfthick, godarr, fldarr, 'img',
                                                    'spl.cfg', objht, over, pupsam, imgsam, 
                                                    psfx, psfy, pixsize, xpix, ypix, aberr, 
                                                    timeout, verbose)
                dataSubGrp.create_dataset('psf', data=psfgrid, dtype=np.uint8)
            else:
                dataSubGrp.create_dataset('psf', shape=(ypix, xpix, 3), dtype=np.uint8) # no actual data is stored
            # CHIEF-RAY INTERSECT DATA for all object planes
            enpp = ln.zGetPupil().ENPP
            expp = ln.zGetPupil().EXPP
            crImgIPtsGrp = dataSubGrp.create_group('cr_img_ipts')
            for thickCnt, thick in enumerate(objsurfthick):
                ln.zSetThickness(surfNum=0, value=thick)
                # push lens into the LDE as array tracing occurs in the LDE
                ln.zPushLens(1)
                x, y, z, err, vig = get_chief_ray_intersects(n=numCrImIpts, real=True)
                crImgIPtsSubGrp = crImgIPtsGrp.create_group('{}'.format(thickCnt).zfill(2))
                crImgIPtsSubGrp.create_dataset('x', data=x, dtype=np.float32)
                crImgIPtsSubGrp.create_dataset('y', data=y, dtype=np.float32)
                crImgIPtsSubGrp.create_dataset('err', data=err, dtype=np.int16)
                set_hdf5_attribs(crImgIPtsSubGrp, {'obj_to_enpp' : enpp + thick}) #
            if verbose:
                print('Traced chief-ray intersects.')
                sys.stdout.flush()
            # set sub-group attribute
            dataSubGrpAttribDict = {'tilt_x' : tiltAbtX,   # others attributes like defocus_waves, paraxial mag etc
                                    'mag' : mag,           # list of magnifications at the different depths
                                    'img_to_expp' : abs(expp)  # Exit pupil to Image plane
                                   }
            set_hdf5_attribs(dataSubGrp, dataSubGrpAttribDict)
    return hdffileFull

#%% Data viewing functions
IMG_DOWN_SAMP_PIX_SKIP = 2

def get_hdf5files_list(stype=1):
    """helper function to get the list of HDF5 files in the `data/imgstack` sub-directory

    Parameters
    ----------
    stype : integer (0/1)
        type of simulation. Use `0` for frontoparallel `1` (default) for lens-tilt

    """
    imgdir = os.path.join(os.getcwd(), 'data', 'imgstack')
    startStr = 'lens_tilt' if stype else 'fronto_para'
    return [f for f in os.listdir(imgdir) if (f.endswith('.hdf5') and f.startswith(startStr))]

def _downsample_img(img):
    """downsample the image data by a factor IMG_DOWN_SAMP_PIX_SKIP
    """
    global IMG_DOWN_SAMP_PIX_SKIP
    return img[::IMG_DOWN_SAMP_PIX_SKIP,::IMG_DOWN_SAMP_PIX_SKIP,:]

def _memmap_ds(h5f, ds):
    """returns a numpy memmap'd array mapped to the data buffer of the dataset `ds`

    Parameters
    ---------- 
    h5f : string 
        HDF5 file name (not the file object)
    ds : object 
        dataset path to be memory-mapped
    
    Assumptions
    ----------- 
    The data is uncompressed and not chunked
    """
    arr = None
    offset = ds.id.get_offset()
    if offset: # offset is "None" if there is no real data 
        arr = np.memmap(h5f, mode='r', shape=ds.shape, 
                    offset=offset, dtype=ds.dtype)
    return arr

def _get_formatted_dat_string(dats):
    """dats : list of magnifications, angles or other data
    """
    datsstr = ['{:2.3f}'.format(each) for each in dats]
    return ', '.join(datsstr)

def show_image_stack(hdffile, i):
    """plots the i-th image in the image stack 
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    with hdf.File(hdffile, 'r') as f:
        dgrp = f['data/'+'{}'.format(i).zfill(3)]
        dsetImg = f['data/'+'{}'.format(i).zfill(3)+'/image']
        img = _memmap_ds(hdffile, dsetImg)
        ax.imshow(_downsample_img(img)) #, interpolation='none')
        #if f.attrs['focal_stack_type'] == 'lenstilts'
        print('Mags:', _get_formatted_dat_string(dgrp.attrs['mag']))
        plt.show()

def show_registered_image_stack(hdffile, i):
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    with hdf.File(hdffile, 'r') as f:
        try:
            #dgrp = f['registered_data/'+'{}'.format(i).zfill(3)]
            dsetImg = f['/registered_data/'+'{}'.format(i).zfill(3)+'/image']
            img = _memmap_ds(hdffile, dsetImg)
        except:
            ax.text(x=0.35, y=0.5, s='NO REGISTERED DATA', fontsize=17)
        else:
            ax.imshow(_downsample_img(img)) #, interpolation='none')
            #if f.attrs['focal_stack_type'] == 'lenstilts'
            #print('Mags:', _get_formatted_dat_string(dgrp.attrs['mag']))
        plt.show()

def show_psf_stack(hdffile, i):
    """helper function to show the i-th PSF in the stack 
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    with hdf.File(hdffile, 'r') as f:
        dgrp = f['/data/'+'{}'.format(i).zfill(3)]
        dsetPsf = f['/data/'+'{}'.format(i).zfill(3)+'/psf']
        psfdat = _memmap_ds(hdffile, dsetPsf)
        if psfdat is not None:
            ax.imshow(_downsample_img(psfdat), interpolation='none')
        else:
            ax.text(x=0.35, y=0.5, s='NO PSF DATA', fontsize=17)
        print('Magnifications:', dgrp.attrs['mag'])
        plt.show()

def show_registered_psf_stack(hdffile, i):
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    with hdf.File(hdffile, 'r') as f:
        try:
            #dgrp = f['rectified_data/'+'{}'.format(i).zfill(3)]
            dsetPsf = f['registered_data/'+'{}'.format(i).zfill(3)+'/psf']
        except:
            ax.text(x=0.35, y=0.5, s='NO REGISTERED DATA', fontsize=17)
        else:
            psfdat = _memmap_ds(hdffile, dsetPsf)
            if psfdat is not None:
                ax.imshow(_downsample_img(psfdat), interpolation='none')
            else:
                ax.text(x=0.35, y=0.5, s='NO PSF DATA', fontsize=17)
                #print('Magnifications:', dgrp.attrs['mag'])
        plt.show()

def _get_zero_lens_tilt_sub_group(f):
    """returns the absolute path to the subgroup that
    corresponds to 0° lens tilt
    
    Parameters
    ----------
    f : object 
        HDF5 file object 
        
    Returns
    -------
    sgrp : HDF5 group object 
        
    Examples
    --------
    >>> subGrp = get_zero_lens_tilt_sub_group(f)
    >>> print(subGrp)
    <HDF5 group "/data/001" (3 members)>
    """
    dataGroups = f['data']
    for sg in dataGroups:
        if(f['data'][sg].attrs['tilt_x'])==0:
            return f['data'][sg]

def show_cr_img_inter_frontoparallel_stack(hdffile, i):
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    with hdf.File(hdffile, 'r') as f:
        dgrp = f['data/'+'{}'.format(i).zfill(3)]
        x = f['data/'+'{}'.format(i).zfill(3)+'/cr_img_ipts/x']
        y = f['data/'+'{}'.format(i).zfill(3)+'/cr_img_ipts/y']
        ax.scatter(x[...], y[...], s=10, facecolors='none', 
                       edgecolors='r', alpha=0.8, zorder=12)
        print('Magnifications:', dgrp.attrs['mag'])
        plt.show()

def show_cr_img_inter_stack(hdffile, i):
    # determine the axes parameters
    g = 0.04    # guard space on left edge
    p = 0.028   # padding between the axes
    w = (1.0 - (1.1*g + 2*p))/3 # width of each axes
    b = 0.4     # bottom of each axes within the figure
    h = 0.6     # height of each axes within the figure
    fig = plt.figure(figsize=(16, 10))
    ax = [fig.add_axes([g, b, w, h]),
          fig.add_axes([g+p+w, b, w, h]),
          fig.add_axes([g+2*(p+w), b, w, h])]

    with hdf.File(hdffile, 'r') as f:
        nx = f.attrs['cr_img_ipts_numx']
        ny = f.attrs['cr_img_ipts_numy']
        dgrp0path = _get_zero_lens_tilt_sub_group(f).name
        dgrp = f['data/' + '{}'.format(i).zfill(3)]
        crImgIptsGrp = f['data/' + '{}'.format(i).zfill(3) + '/cr_img_ipts']
        xlim, ylim = 0, 0
        for distCnt, grp in enumerate(crImgIptsGrp):
            x0 = f[dgrp0path + '/cr_img_ipts/' + '{}'.format(distCnt).zfill(2) + '/x'] 
            y0 = f[dgrp0path + '/cr_img_ipts/' + '{}'.format(distCnt).zfill(2) + '/y']
            x = crImgIptsGrp[grp]['x']
            y = crImgIptsGrp[grp]['y']
            # plot intersection points for no lens tilt ... this sets the figure parameters
            ax[distCnt].scatter(x0, y0, marker='o', s=15, facecolors='none', 
                       edgecolors='b', alpha=0.8, zorder=14)
            x0GridPts = x0[:nx]
            y0GridPts = y0[::ny] 
            ax[distCnt].vlines(x0GridPts, ymin=min(y0GridPts), ymax=max(y0GridPts), 
                      zorder=2, colors='#CFCFCF', lw=0.8)
            ax[distCnt].hlines(y0GridPts, xmin=min(x0GridPts), xmax=max(x0GridPts), 
                      zorder=2, colors='#CFCFCF', lw=0.8)
            # plot the intersection points for non-zero lens-tilts
            ax[distCnt].scatter(x, y, marker='x', s=40, c='r', alpha=0.9, zorder=15)
            # quiver plot
            M = np.hypot(x[:]-x0[:], y[:]-y0[:])
            Q = ax[distCnt].quiver(x0, y0, x[:]-x0[:], y[:]-y0[:], M, 
                          units='xy', pivot='tail', width=0.05,
                          scale=1, alpha=0.8, zorder=12)
            
            cax, _ = make_axes(ax[distCnt], orientation='horizontal', 
                                 fraction=0.01, anchor=(0.0, 1.0),
                                 pad=0.05, shrink=1, aspect=90)
            cb = plt.colorbar(mappable=Q, format='%.3f', ticks=plt.MaxNLocator(6),
                              cax=cax, orientation='horizontal')
            cb.solids.set_edgecolors("face")
            cb.solids.set_rasterized(True)
            pad = 1.1*M.max() or 0.25
            xlimAx = np.max(np.abs(x0)) + pad
            ylimAx = np.max(np.abs(y0)) + pad
            xlim = xlim if xlim > xlimAx else xlimAx
            ylim = ylim if ylim > ylimAx else ylimAx
            ax[distCnt].set_aspect('equal')
            ax[distCnt].set_title('Obj to ENPP : {} mm'.format(crImgIptsGrp[grp].attrs['obj_to_enpp']))
        for distCnt in range(3):
            ax[distCnt].set_xlim(-xlim, xlim)
            ax[distCnt].set_ylim(-ylim, ylim)
        #fig.tight_layout() causes eror
        fig.text(0.009, 0.7, r'$\bf{y}\,\it{(mm)}$', fontsize=17, rotation='vertical')
        fig.text(0.5, 0.35, r'$\bf{x}\,\it{(mm)}$', fontsize=17, rotation='horizontal')
        tiltx = f['data/' + '{}'.format(i).zfill(3)].attrs['tilt_x']
        fig.text(0.45, 0.965, 
                 r'$\alpha_x, \alpha_y = {}^o,{}^o$'.format(tiltx, 0.0), fontsize=19)
        
        print('Magnifications:', dgrp.attrs['mag'])

iSelect = None # hack for now

def show_stack(hdffile, what):
    global iSelect
    if iSelect:
        iSelect.close()
    imgdir = os.path.join(os.getcwd(), 'data', 'imgstack')
    hdffile = os.path.join(imgdir, hdffile)
    with hdf.File(hdffile, 'r') as f:
        stackLen = len(f['data'])
        iSelect = widgets.IntSlider(value=0, min=0, max=stackLen-1, step=1, 
                                    description='ImageNum ({})'.format(stackLen), 
                                    orientation='horizontal')
        interact(what, hdffile=fixed(hdffile), i=iSelect)

#%% Data processing and registration

def _normalize_2D_pts(p):
    """Function to normalize 2D homogeneous points
    
    This function, which is used for pre-conditioning 2D homogeneous points 
    before solving for homographies and fundamental matrices, translates and
    normalizes the set of points so that their centroid is at the origin, and
    their mean distance from the origin is sqrt(2)
    
    Parameters
    ----------
    p : ndarray
        ``p`` is a `3xN` array of for the set of `N` 2D homogeneous points
        
    Returns
    -------
    newPts : ndarray
        ``newPts`` has the same shape as ``p`` after normalization. Specifically,
        `newPts = np.dot(T, p)`
    T : ndarray
        the 3x3 similarity transformation matrix     
    """
    eps = np.spacing(1)
    finiteindex = np.where(np.abs(p[-1]) > eps)[0]

    # enforce the scale to be 1 for all finite points    
    p[:, finiteindex] = p[:, finiteindex]/p[-1, finiteindex]
    
    c = np.mean(p[:2, finiteindex], axis=1)      # centroid of finite points
    pNew = p[:2, finiteindex] - c[:, np.newaxis] # shift origin to centroid
    dist = np.sqrt(np.sum(pNew**2, axis=0)) 
    scale = np.sqrt(2)/np.mean(dist)
    T = np.diag([scale, scale, 1.0])
    T[:2, 2] = -c*scale
    return np.dot(T, p), T

def _get_homography2D(fp, tp, method='DLT', normbyh9=True):
    """Return the homography ``H``, such that ``fp`` is mapped to ``tp`` using 
    normalized DLT described in Algorithm (4.2) of Hartley and Zisserman. 
    
    Parameters
    ----------
    fp : ndarray
        ``fp`` can be a ``2xN`` or ``3xN`` ndarray of "from"-points. If ``fp`` is 
        ``3xN`` the scaling factors ``w_i`` may or may not be 1. i.e the structure 
        of ``fp = _np.array([[x0, x1, ...], [y0, y1, ...], [w0, w1, ...]])``. 
        If ``fp`` is 2xN, then it is assumed that ``w_i = 1`` in homogeneous
        coordinates. i.e. ``fp = _np.array([[x0, x1, ...], [y0, y1, ...]])``
    tp : ndarray
        a ``2xN`` or ``3xN`` ndarray of corresponding "to"-points. If ``tp`` is 
        ``3xN`` the scaling factors ``w_i'`` may or may not be 1. i.e the structure 
        of ``tp = _np.array([[x0', x1', ...], [y0', y1', ...], [w0', w1', ...]])``. 
        If ``tp`` is 2xN, then it is assumed that ``w_i' = 1`` in homogeneous 
        coordinates is 1. i.e. ``tp = _np.array([[x0', x1', ...], [y0', y1', ...]])``
    method : string, optional
        method to compute the 2D homography. Currently only normalized DLT has
        been implemented
    normbyh9 : bool, optional
        if ``True`` (default), the homography matrix ``H`` is normalized by 
        dividing all elements by ``H[-1,-1]``, so that ``H[-1,-1] = 1``. However, 
        this normalization will fail if ``H[-1,-1]`` is very small or zero (if
        the coordinate origin is mapped to a point at infinity by ``H``)
    
    Returns
    -------
    H : ndarray
        the 3x3 homography, ``H`` such that ``tp = np.dot(H, fp)``
    """
    if fp.shape != tp.shape:
        raise RuntimeError("The point arrays must have the same shape!")
    if (fp.shape[0] < 2) or (fp.shape[0] > 3):
        raise RuntimeError("The length of the input arrays in the first "
                           "dimension must be 3 or 2")
    numCorrespondences = fp.shape[1]
    if fp.shape[0] == 2:
        fp = np.r_[fp, np.ones((1, numCorrespondences))]
        tp = np.r_[tp, np.ones((1, numCorrespondences))]

    fp, T = _normalize_2D_pts(fp)
    tp, Tdash = _normalize_2D_pts(tp)

    # create matrix A of size 2*N by 9
    A = np.zeros((2*numCorrespondences, 9))
    wdash = tp[2,:].tolist()
    for i in range(numCorrespondences):
        x = fp[:,i]
        xdash = tp[:2, i]
        A[2*i:2*(i+1), :] = np.kron(np.c_[np.eye(2)*wdash[i], -xdash], x)
    
    # The solution is the unit singular vector corresponding to the smallest
    # singular value of A
    U, S, Vh = np.linalg.svd(A)
    Htilde = Vh[8,:].reshape((3,3))
    
    # Denormalization H = T'^-1 H_tilde T
    H = np.dot(np.linalg.inv(Tdash), np.dot(Htilde, T))    

    if normbyh9:
        H = H/H[2,2]
    return H

def _get_homography_from_CR_intersects(f, tiltCnt=0):
    """returns the homographies between all object-planes and the 
    single image plane for the tilted (lens) configuration `tiltCnt`. 
    
    The homographies are computed from deviation of the chief ray 
    intersects for the given tilt. The unit associated with the values
    are in physical units (lens unit, mostly mm)
    
    Parameters
    ----------
    f : object
        hdf5 file handle
    tiltCnt : integer 
        tilted configuration number  
        
    Returns
    -------
    Hcr : ndarray 
        a (3, 3, n) ndarray where `n` represents the object plane number
    """
    pixsize = f.attrs['img_sim_pixel_size']
    noTiltGrp = _get_zero_lens_tilt_sub_group(f).name
    crImgIptsGrp = f['data/' + '{}'.format(tiltCnt).zfill(3) + '/cr_img_ipts']
    # for all the object planes
    Hstack = []
    for distCnt, grp in enumerate(crImgIptsGrp):
        xRef = f[noTiltGrp + '/cr_img_ipts/' + '{}'.format(distCnt).zfill(2) + '/x'] 
        yRef = f[noTiltGrp + '/cr_img_ipts/' + '{}'.format(distCnt).zfill(2) + '/y']
        x = crImgIptsGrp[grp]['x']
        y = crImgIptsGrp[grp]['y']
        fp = np.vstack((xRef, yRef))/pixsize
        tp = np.vstack((x, y))/pixsize
        H = _get_homography2D(fp, tp)
        Hstack.append(H)
    return np.stack(Hstack, axis=2)


def _get_registered_data(f, tiltCnt, method='crii'):
    """register images and psf grid using `method`. 

    Currently only `crii` (chief-ray image-plane intersects) is used

    Parameters
    ---------- 
    f : object 
        HDF5 file object 
    tiltCnt : integer 
        the integer position number of the subgroup in the HDF5 file 
        containg the `image`, `psf` and `cr_img_ipts datum` for the particular 
        tilted orientation of the lens. 
    method : string 
        'crii' using chief-ray intersects in the image plane, supported currently 

    Returns
    ------- 
    regImg : ndarray
        registered image  
    regPsf : ndarray or None
        registered PSF grid (if PSF grid data exists) 
    H : ndarray 
        hompgraphy 
    """
    Hcr = _get_homography_from_CR_intersects(f, tiltCnt)
    # TO DO ;; else other logic 
    # test that the 3 homographies computed between the three object planes
    # and the image plane are equal for all configurations of the lens tilt
    assert (np.allclose(Hcr[:,:,0], Hcr[:,:,1]) and 
            np.allclose(Hcr[:,:,0], Hcr[:,:,2]))
    H = Hcr[:,:,0].copy()
    # register image
    dset = f['data/'+'{}'.format(tiltCnt).zfill(3)+'/image']
    img = _memmap_ds(f.filename, dset)
    rows, cols, _ = img.shape
    ## perspective warp
    regImg = cv2.warpPerspective(img, H, (cols, rows))
    ## register psf
    regPsf = None
    dset = f['data/'+'{}'.format(tiltCnt).zfill(3)+'/psf']
    img = _memmap_ds(f.filename, dset)
    if img: # Not all files have actual PSF data
        regPsf = cv2.warpPerspective(img, H, (cols, rows))
    return regImg, regPsf, H


def register_data(hdffile):
    """register the images and psf grid in the file `hdffile`

    The function creates a group called `registered_data` under the root 
    group. The registered image, psf grid data, and corresponding homographies
    are stored under subgroups with names as three digit numbers 

    Parameters
    ---------- 
    hdffile : string 
        file name with absolute path to the HDF5 file 

    Returns
    ------- 
    None 

    """
    with hdf.File(hdffile, 'r+') as f:
        stackLen = len(f['data'])
        if not 'registered_data' in f.keys():
            grp = f.create_group('registered_data')
            for tiltCnt in range(stackLen):
                subGrp = grp.create_group('{}'.format(tiltCnt).zfill(3))
                regImg, regPsf, H = _get_registered_data(f, tiltCnt, method='crii')
                tiltx = f['data/'+'{}'.format(tiltCnt).zfill(3)].attrs['tilt_x']
                subGrp.create_dataset('image', data=regImg, dtype=np.uint8)
                if regPsf:
                    subGrp.create_dataset('psf', data=regPsf, dtype=np.uint8)
                else:
                    subGrp.create_dataset('psf', shape=regImg.shape, dtype=np.uint8)
                subGrp.create_dataset('H', data=H, dtype=np.float64)
                set_hdf5_attribs(subGrp, {'tilt_x': tiltx})
            print('OK! Registered data and embedded into HDF file.')
        else:
            print('Registered data already in file. Did not re-register!')
    

def save_unregistered_images(hdffile, savedir):
    with hdf.File(hdffile, 'r') as f:
        stackLen = len(f['data'])
        print('No. of images:', stackLen)
        for tiltCnt in range(stackLen):
            dset = f['data/'+'{}'.format(tiltCnt).zfill(3)+'/image']
            img = _memmap_ds(f.filename, dset)
            imgname =  '{}'.format(tiltCnt).zfill(3) + '.png'
            imgfilename = os.path.join(savedir, imgname)
            imsave(imgfilename, img)
    print('OK')

def save_registered_images(hdffile, savedir):
    with hdf.File(hdffile, 'r') as f:
        stackLen = len(f['registered_data'])
        print('No. of images:', stackLen)
        for tiltCnt in range(stackLen):
            dset = f['registered_data/'+'{}'.format(tiltCnt).zfill(3)+'/image']
            img = _memmap_ds(f.filename, dset)
            imgname =  '{}'.format(tiltCnt).zfill(3) + '.png'
            imgfilename = os.path.join(savedir, imgname)
            imsave(imgfilename, img)
    print('OK')
        

if __name__ == '__main__':
    pass