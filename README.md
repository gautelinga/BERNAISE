# BERNAISE
### Binary ElectRohydrodyNAmIc SolvEr
BERNAISE is a flexible, high-level solver of electrohydrodynamic flows in complex geometries currently under development.
It is written in Python and built on the FEniCS project, which in turn effectively interfaces to optimized linear algebra backends such as PETSc.

### Work plan

* Time dependent EHD
* Time dependent PF for the two-phase flow  
* Add the two above together
* More complicated geometries
* **Nobel Prize!!!** ***(only non-optional)***

### Folder plan
         
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

### Master Minds: 
* **Gaute** er en ***nordmand***.
* **Asger** er ***dansk***.
* **Joachim** er *også* ***dansk***.
* **Alfred** *var* en ***svensker***.
