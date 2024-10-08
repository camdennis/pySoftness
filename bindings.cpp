#include <pybind11/pybind11.h>
#include "structureFactor.cpp"
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

//extern "C" void getRij(double* rij, double* positions, double L, double gamma, int N);


PYBIND11_MODULE(libstructureFactor, m) {
    m.doc() = "A module to calculate the potential";

    m.def("getGR", [](py::array_t<double> gr, py::array_t<double> rList, py::array_t<double> positions, int numR, double L, double gamma, double rho, double delta, int NA, int NB) {
        py::buffer_info gr_buf = gr.request();
        py::buffer_info rList_buf = rList.request();
        py::buffer_info positions_buf = positions.request();

        double* gr_ptr = static_cast<double*>(gr_buf.ptr);
        double* rList_ptr = static_cast<double*>(rList_buf.ptr);
        double* positions_ptr = static_cast<double*>(positions_buf.ptr);
        getGR(gr_ptr, rList_ptr, positions_ptr, numR, L, gamma, rho, delta, NA, NB);
    }),
    m.def("getRij", [](py::array_t<double> rij, py::array_t<double> positions, double L, double gamma, int N) {
        py::buffer_info rij_buf = rij.request();
        py::buffer_info positions_buf = positions.request();

        double* rij_ptr = static_cast<double*>(rij_buf.ptr);
        double* positions_ptr = static_cast<double*>(positions_buf.ptr);
        getRij(rij_ptr, positions_ptr, L, gamma, N);
    }),
    m.def("getISFOverlap", [](py::array_t<double> ISFOverlap, py::array_t<double> positions1, py::array_t<double> positions2, double L, double gamma, double rmax, double cutoff, int N) {
        py::buffer_info positions1_buf = positions1.request();
        py::buffer_info positions2_buf = positions2.request();
        py::buffer_info ISFOverlap_buf = ISFOverlap.request();

        double* positions1_ptr = static_cast<double*>(positions1_buf.ptr);
        double* positions2_ptr = static_cast<double*>(positions2_buf.ptr);
        double* ISFOverlap_ptr = static_cast<double*>(ISFOverlap_buf.ptr);
        getISFOverlap(ISFOverlap_ptr, positions1_ptr, positions2_ptr, L, gamma, rmax, cutoff, N);
    });
}