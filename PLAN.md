

### Work plan of 31/8/2017 
* Big Steps
  * i) Make Unit conversion utility (Asger)
  * ii) Improved solver -> Iterativ solver with some preconditioner (Gaute)
  * iii) quasi-3D solvers. First a 2D-depth average for simunlating flow between two plates
  * iv) Eletric intial geuss for for intrusion either slove some modified Poisson-Boltzmann equation or solve the electic system in the intial condition for a couple of "time" iterations before begening to solve PF and NS
  * v) Viscus fingering (a la M. Bazant) 
  * vi) Full 3D simulations for that point ii) have to be solved and a effort at point iii) of making a axial symetric solver have to be made so the results can be beched-marked and tested in a porper way

* Small and unfinsihed busisness 
  * Convegens sudies of the intrusion_bulk.py problem
  * Other Convergens tests
  * Some kind of bech mark... maybe a manifucted solution  

### Work plan form 8/6/2017 

* a. Things that have to work in order to continue.
  * do riesing charged bobubel by electric force!
  * taylor-hood -> grave(or iterative solver). replace with 1 order elemets  
  * Find the equations, dimensionless. And define parameter range!
  * establish convergence
    
* b. tests
  * Raylay-Taylor PF and NS
  * advective front stable/ustable PF and NS 
  * Debye layer EC (and NS)
  * Coulomb law EC (and PF)

* c. benchmarks
  * riesing charged bobubel 
  * collinding opposid charged bobubels
  * capilar introsion
  * diaelctric breakdown  

### Work plan from the beginng: 
* i. Formulate the dimensionless equation, find the equations, then do the following:  
       * X Do it for Nernst-Planck
       * X Do it for Poisson
       * X Do it for Stokes
       * - Do it for the phase field 

*  1. Phase Field (Gaute)
    There is properly a way to do this!
       * - Make input for velosity 
       * - add maybe contact angel
       * - make unit test for stright capilar   
       *  - make unit test for barbell capilar 

* 2. Mesh genration (Asger) stright forward!
        * X Make stright capilar 
        * X Make the barbell capilar
        * X Make mesh load function 
        * - Make unit test 

* 3. Time-dependent Stokes straight forward!
       * - Make space for both electic force, interface force and spacedependet viscosity(the two phases)
       * - make unit test for stright capilar   
       * - make unit test for barbell capilar  

* 4. Stady-state Poisson-Nernst-Plack (Asger)a bit invold but have som legacy code!
       * - Make it woke with spacedependet permativity and spacedependet defusion constats (the two phases)
       * - make unit test for barbell capilar  

* 5. Timedependet Poisson Nerst-Plack. A new thing pu proberly ok to deal with!
       * - Make it woke with spacedependet permativity and spacedependet defusion constats (the two phases)
       * - make unit test for barbell capilar  

* 6. Put it all togeter! A bit diffcault to say at present how that is gonna go dwon!
       * - Make config file and sauce.py work 