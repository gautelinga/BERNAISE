//cppimport
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <eigen3/Eigen/Eigen>
#include <random>
#include <iostream>
#include <algorithm>
#include <set>

namespace py = pybind11;

typedef Eigen::Vector2d Vector;

struct less_than_key
{
    inline bool operator() (const Vector& v1, const Vector& v2)
    {
        return (v1[0] < v2[0] || (v1[0] == v2[0] && v1[1] < v2[1]));
    }
};


// Passing in a generic array
py::array_t<double> place_obstacles(std::size_t N, double Lx, double Ly, double R, 
                                    double pad_x, double pad_y, int seed, size_t max_tries)
{
    const size_t gdim = 2;
    //const size_t max_tries = 1000000;
    const double D2 = 4*R*R;

    if (seed == -1){
        std::random_device rd;
        seed = rd();
    }
    std::mt19937 gen(seed);
    
    std::uniform_real_distribution<> uniform_x(-Lx/2+pad_x, Lx/2-pad_x);
    std::uniform_real_distribution<> uniform_y(-Ly/2+pad_y, Ly/2-pad_y);

    std::vector<Vector> x(N);
    Vector L = {Lx, Ly};

    bool found = false;

    // cache map
    size_t nxgrid = 100;
    size_t nygrid = 200;
    double dx = Lx/nxgrid;
    double dy = Ly/nygrid;

    std::vector<std::set<size_t>> ids;
    ids.resize(nxgrid*nygrid);

    size_t i = 0;
    for (; i < N; ++i)
    {
        found = false;
        size_t tries = 0;
        Vector xnew = {0., 0.};

        // size_t jx_min, jy_min, jx_max, jy_max;

        while ( !found && tries < max_tries ) 
        {
            xnew = {uniform_x(gen), uniform_y(gen)};
            found = true;
            if (i == 0)
            {
                break;
            }
    
            /*
            jx_min = floor(xnew[0]-R+Lx/2)/dx;
            jy_min = floor(xnew[1]-R+Ly/2)/dy;
            jx_max = floor(xnew[0]+R+Lx/2)/dx;
            jy_max = floor(xnew[1]+R+Ly/2)/dy;

            // std::cout << jx_min << " " << jx_max << " " << jy_min << " " << jy_max << std::endl;
            std::set<size_t> ids_loc;

            for (size_t kx=jx_min; kx <= jx_max; ++kx)
            {
                size_t kx_mod = (kx + nxgrid) % nxgrid;
                for (size_t ky=jy_min; ky <= jy_max; ++ky)
                {
                    size_t ky_mod = (ky + nygrid) % nygrid;
                    size_t pos = ky_mod * nxgrid + ky_mod;
                    //ids_loc.insert(ids[pos].begin(), ids[pos].end());
                    for ( auto & id : ids[pos] ){
                        ids_loc.insert(id);
                    }
                }
            }

            for ( auto & id : ids_loc ){
                Vector dx = x[id] - xnew;
                for (size_t d = 0; d < gdim; ++d){
                    dx[d] = abs(dx[d]);
                    dx[d] = std::min(dx[d], abs(L[d]-dx[d]));
                }
                if (dx.squaredNorm() < D2)
                {
                    found = false;
                    break;
                }
            }
            */

            for (size_t j=0; j < i; ++j)
            {
                Vector dx = x[j] - xnew;
                for (size_t d = 0; d < gdim; ++d)
                {
                    dx[d] = abs(dx[d]);
                    dx[d] = std::min(dx[d], abs(L[d]-dx[d]));
                }

                if (dx.squaredNorm() < D2)
                {
                    found = false;
                    break;
                }
            }
            ++tries;
        }
        if (found)
        {
            /*
            for (size_t kx=jx_min-1; kx <= jx_max+1; ++kx)
            {
                size_t kx_mod = (kx + nxgrid) % nxgrid;
                for (size_t ky=jy_min-1; ky <= jy_max+1; ++ky)
                {
                    size_t ky_mod = (ky + nygrid) % nygrid;
                    ids[ky_mod * nxgrid + ky_mod].insert(i);
                }
            }
            */

            std::cout << "Found " << i << " after " << tries << " tries." << std::endl;
            x[i] = xnew;
        }
        else 
        {
            break;
        }
    }

    if (!found)
    {
        N = i;
    }
    x.resize(N);
    std::sort(x.begin(), x.end(), less_than_key());

    py::array_t<double> result = py::array_t<double>( N * gdim );
    py::buffer_info buf = result.request();
    double *ptr = static_cast<double *>(buf.ptr);

    for (size_t idx=0; idx < N; ++idx)
    {
        for (size_t d=0; d<gdim; ++d)
        {
            ptr[gdim*idx + d] = x[idx][d];
        }
    }
    result.resize({N, gdim});

    return result;
}

PYBIND11_MODULE(rsa, m) {
    m.def("place_obstacles", &place_obstacles, 
    py::arg("N"), py::arg("Lx"), py::arg("Ly"), py::arg("R"),
    py::arg("pad_x")=0., py::arg("pad_y")=0.,
    py::arg("seed")=-1, py::arg("max_tries")=10000000);
}

<%
cfg['compiler_args'] = ['-O3', '-march=native'] 
setup_pybind11(cfg)
%>