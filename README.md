# BERNAISE
[![Build Status](https://travis-ci.org/gautelinga/BERNAISE.svg?branch=master)](https://travis-ci.org/gautelinga/BERNAISE)
 _BERNAISE_ (Binary ElectRohydrodyNAmIc SolvEr) is a flexible, high-level solver of electrohydrodynamic flows in complex geometries currently under development.
It is written in Python and built on the FEniCS project, which in turn effectively interfaces to optimized linear algebra backends such as PETSc.

<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/droplet.gif" width=122 height=254 alt="Buoyancy-driven droplet"/>
    <br /><b>Buoyancy-driven droplet.</b>
</p>
<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/charged_droplets.gif" width=264 height=87 alt="Colliding oppositely charged droplets"/><br />
    <b>Two colliding oppositely charged droplets.</b> Red: positive charge, blue: negative charge.
</p>
<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/dielectric_faster.gif" width=192 height=192 alt="Two-phase dielectricum."/><br />
    <b>Two-phase dielectricum/capacitor.</b> Red: positive charge, blue: negative charge. Top: negative surface charge, bottom: positive surface charge.
</p>
<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/snoevsen.gif" width=250 height=140 alt="Snøvsen."/>
    <img src="http://www.nbi.dk/~linga/bernaise/snoevsen_neutral.gif" width=250 height=140 alt="Snøvsen, neutral."/><br />
    <b>Enhanced oil recovery</b> by application of a surface charge to the pore wall, and ions dissolved in the water phase.
    The color indicates the charge.
    The flow is driven by a constant velocity at the top (Couette flow).
    <b>Left:</b> With (uniform) surface charge, the droplet is released into the bulk.
    <b>Right:</b> Without surface charge, the droplet stays within the pore.
    Note that the droplet is slightly asymmetric due to the imposed flow.
</p>

<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass_pore/p0cm10.gif" width=262 height=87 alt="Hourglass with surface charge and zero bias pressure"/>
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass_pore/p5cm10.gif" width=262 height=87 alt="Hourglass with surface charge and small bias pressure"/>
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass_pore/p50cm10.gif" width=262 height=87 alt="Hourglass with surface charge and large bias pressure"/><br />
</p>
<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass_pore/p0c0.gif" width=262 height=87 alt="Hourglass without  surface charge and zero bias pressure"/>
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass_pore/p5c0.gif" width=262 height=87 alt="Hourglass without surface charge and small bias pressure"/>
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass_pore/p50c0.gif" width=262 height=87 alt="Hourglass without surface charge and large bias pressure"/><br />
    <b>Enhanced oil recovery</b> in a pore throat by application of a surface charge to the pore wall, and ions dissolved in the water phase.
    The color indicates the charge (as above).
    In the four figures to the right, the flow is driven by a pressure difference; in the two to the left there is zero pressure difference between the two sides.
    <b>Upper:</b> With (uniform) surface charge in the throat, the droplet is released into the bulk even without external forcing.
    <b>Lower:</b> Without surface charge, the droplet stays within the pore, except for large external forcing.
</p>

<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/flipper.gif" width=197 height=165 alt="A dolphin being cleaned from oil spill."/><br />
    <b>Animal decontamination:</b> A dolphin initially immersed in oil is fully cleaned by the application of surface charge to the dolphin's skin, and ions in the water.
    Red: positive charge, blue: negative charge.
</p>

### Features
* Simulates time-dependent two-phase electrohydrodynamics in 2D using a phase-field approach.
* Supports complex geometries represented by unstructured meshes.
* Easy implementation of new problems and solvers.

### Planned features (but currently not supported)
* Adaptive time-stepping.
* Two-phase EHD in 3D.
* For planed work look in PLAN.md 
* **Nobel Prize!!!** ***(only non-optional)***

### Folder structure
```
* BERNAISE
  * common
    * __init__.py
    * bcs.py
    * cmd.py
    * functions.py
    * io.py
    * recipe.txt
  * problems
    * __init__.py
    * simple.py
    * ...
  * solvers
    * __init__.py
    * basic.py
    * ...
  * tests
    * ...
  * meshes
    * ...
  * utilities
    * extract_polygon.py
    * generate_mesh.py
    * load_mesh.py
    * plot.py
    * units.py
  * documentation
    * ...
  * articles
    * ...
  * README.md
  * sauce.py
  * postprocess.py
```

### Dependencies
* FEniCS/Dolfin
* fenicstools (for post-processing)
* simplejson
* mpi4py
* h5py (parallel)
* numpy
* skimage (for polygon extraction tool)
* tabulate (for post-processing)

### Master Minds: 
* **Gaute** er en ***nordmand***.
* **Asger** er ***dansk***.
* **Joachim** er *også* ***dansk***.
* **Alfred** *var* en ***svensker***.
