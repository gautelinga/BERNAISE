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
    <b>Enhanced oil recovery</b> by application of a surface charge to the pore wall, and ions dissolved in the water phase. The color indicates the charge. The flow is driven by a constant velocity at the top (Couette flow). <b>Left:</b> With (uniform) surface charge, the droplet is released into the bulk. <b>Right:</b> Without surface charge, the droplet stays within the pore. Note that the droplet is slightly asymmetric due to the imposed flow.
</p>

<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass/p0cm10.gif" width=200 height=100 alt="Hourglass with surface charge and zero bias pressure"/>
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass/p5cm10.gif" width=200 height=100 alt="Hourglass with surface charge and small bias pressure"/>
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass/p50cm10.gif" width=200 height=100 alt="Hourglass with surface charge and large bias pressure"/><br />
</p>
<p align="center">
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass/p0c0.gif" width=200 height=100 alt="Hourglass without  surface charge and zero bias pressure"/>
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass/p5c0.gif" width=200 height=100 alt="Hourglass without surface charge and small bias pressure"/>
    <img src="http://www.nbi.dk/~linga/bernaise/hourglass/p50c0.gif" width=200 height=100 alt="Hourglass without surface charge and large bias pressure"/><br />
    <b>Enhanced oil recovery</b> by application of a surface charge to the pore wall, and ions dissolved in the water phase. The color indicates the charge. The flow is driven by a pressure in the four to the right the two to the left there is zero pressure diffence of the two sides. <b>Upper:</b> With (uniform) surface charge in the throat, the droplet is released into the bulk even whitout extreal forcing. <b>Lower:</b> Without surface charge, the droplet stays within the pore, except for large external forcing. Note that the droplet is slightly asymmetric due to the imposed flow.
</p>

### Work plan

* Time dependent EHD
* Time dependent PF for the two-phase flow  
* Add the two above together
* More complicated geometries
* **Nobel Prize!!!** ***(only non-optional)***

### Folder plan
```
* BERNAISE
  * common
    * __init__.py
    * io.py
    * cmd.py
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
    * meshgeneration.py
    * meshload.py
    * ...
  * documentation
    * ...
  * articles
    * ...
  * README.md
  * sauce.py
```

### Master Minds: 
* **Gaute** er en ***nordmand***.
* **Asger** er ***dansk***.
* **Joachim** er *også* ***dansk***.
* **Alfred** *var* en ***svensker***.
