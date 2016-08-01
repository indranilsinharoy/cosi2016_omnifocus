# Omnifocus Image Synthesis using Lens Swivel

Repository for sharing code and simulation results from my research related to 
the paper "Omnifocus image synthesis using lens swivel," I. Sinharoy, P. Rangarajan, 
and M. Christensen, Imaging and Applied Optics 2016, OSA. 


## Contents of the repository

### Computational notebooks (Jupyter)

* [Zemax based simulation of omnifocus image synthesis using lens swivel](http://htmlpreview.github.io/?https://github.com/indranilsinharoy/cosi2016_omnifocus/blob/master/html/omnifocus_simulation.html) - documents the simulation process, plots imtermediate results, and other essential information about the simulation setup.

### Local modules

* `omnifocuslib` - contains several utility functions (over 45 functions) for simulation of 
   omnifocus image synthesis using Zemax and PyZDDE. Several functions are included for 
   geometric optics computations, automating the simulation, storing and retrieving image 
   stack data into and from tagged HDF5 format, computing homographies, etc. 

### Datasets

* fronto_para_focal_stack_2016_07_31_02_10.hdf5 - image stack generated for testing initial code using frontoparallel focal stacking.
* lens_tilt_focal_stack_2016_03_21_02_19.hdf5 - image stack containing the simulated images obtained under lens rotation, registered images and homographies. This dataset was used in producing the results in the paper. 

The HDF5 files are in the directory `\data\imgstack`. 

### Zemax files

I have included two Zemax lens files that used in the computational notebook. Especially, the file 
`paraxialDoubleLens24mmFiniteConj_mp1_cardinalsDrawnWdRotAbtENPP.zmx` was used for the simulation presented in the paper. 






For a discourse on the topic, please visit https://indranilsinharoy.com/projects/omnifocus/


**PLEASE NOTE:** The computational notebooks has not been "cleaned up" in its current form. 
It contains some of the experimentations and explorations. 
I expect to cleanup the notebook soon.


## License

The code is under the `MIT License <http://opensource.org/licenses/MIT>`__.

Note that I don't guarantee the accuracy or performance of the code in the repository. I 
have tried to ensure that the code is accurate to the best of my ability. Also, I haven't
paid much attention on improving the performance of the algorithms. 