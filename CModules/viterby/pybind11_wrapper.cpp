
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../PyToC/pyarrayCasts.hpp"
#include "Viterby_deform.hpp"


//PYBIND11_MAKE_OPAQUE(std::vector<float>);

//using floatVec=std::vector<float>;

PYBIND11_MODULE(viterby, m) {
    m.doc() = "Viterby Algorithm to deform a line";
    m.def("fit_line", &getBestPath, "A function that determines the best deformation to fit a line to another line using a matching score");
    m.def("fit_line_bayes", &getBestPathBayes, "A function that determines the best deformation to fit a line to another line using probabilities");
    m.def("fit_line_and_channel", &getBestPathAndChannel, "A function that determines the best deformation to fit a line to another line and selects the optimal channel");
}