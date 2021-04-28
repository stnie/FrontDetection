
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../PyToC/pyarrayCasts.hpp"
#include "calculateVariables.hpp"




PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin";
    m.def("ept", &calculateEquivalentPotentialTemperature<float>, "A function that prints some stuff");
    m.def("ept", &calculateEquivalentPotentialTemperature<double>, "A function that prints some stuff");
}