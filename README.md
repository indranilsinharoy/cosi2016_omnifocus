# Omnifocus Image Synthesis using Lens Swivel

Repository for sharing code and simulation results from my research related to 
the paper "Omnifocus image synthesis using lens swivel," I. Sinharoy, P. Rangarajan, 
and M. Christensen, Imaging and Applied Optics 2016, OSA. 


[![DOI](https://zenodo.org/badge/3811/indranilsinharoy/cosi2016_omnifocus.svg)](https://zenodo.org/badge/latestdoi/3811/indranilsinharoy/cosi2016_omnifocus)



## Contents of the repository

### Computational notebooks (Jupyter)

* [Zemax based simulation of omnifocus image synthesis using lens swivel](https://dl.dropboxusercontent.com/u/20104715/github/cosi2016_omnifocus/omnifocus_simulation.html) - documents the simulation process, plots imtermediate results, and other essential information about the simulation setup.

### Local modules

* `omnifocuslib` - contains several utility functions (over 45 functions) for simulation of 
   omnifocus image synthesis using Zemax and PyZDDE. Several functions are included for 
   geometric optics computations, automating the simulation, storing and retrieving image 
   stack data into and from tagged HDF5 format, computing homographies, etc. 

### Datasets

* fronto_para_focal_stack_2016_07_31_02_10.hdf5 - image stack generated for testing initial code using frontoparallel focal stacking.
* lens_tilt_focal_stack_2016_03_21_02_19.hdf5 - image stack containing the simulated images obtained under lens rotation, registered images and homographies. This dataset was used in producing the results in the paper. 

Due to file size restrictions of GitHub, the above files are not part of the repository. However, I have uploaded them to dropbox and provided instructions in the Jupyter notebook on how to download them.

The HDF5 files should in the directory `\data\imgstack`. 

### Zemax files

I have included two Zemax lens files that used in the computational notebook. Especially, the file 
`paraxialDoubleLens24mmFiniteConj_mp1_cardinalsDrawnWdRotAbtENPP.zmx` was used for the simulation presented in the paper. 




## 


For a discourse on the topic, please visit https://indranilsinharoy.com/projects/omnifocus/


**PLEASE NOTE:** The code in this repository is not meant to be *production-ready* code. It is essentially *research code* (whose main purpose is experimentation and proof of concept). Also, there are few things that still needs to be done---which I have clearly marked using **<font color='magenta'>To do!</font>** and provided comments.   



## License

The code is under the [MIT License](http://opensource.org/licenses/MIT).

Note that I don't guarantee the accuracy or performance of the code in the repository. I 
have tried to ensure that the code is accurate to the best of my ability. Also, I haven't
paid much attention on improving the performance of the algorithms. 