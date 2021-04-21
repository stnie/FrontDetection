#pragma once

#include <vector>
#include "../PyToC/pyarrayCasts.hpp"
#include <pybind11/numpy.h>


template<class T>
simpleMatrix<T> sphereDistanceTransform_(simpleMatrix<T>&& map, simpleVector<T>&& stretch){



    // THINGY GOES HERE Yays
    T* data = new T[2<<29];
    simpleMatrix<T> map2{data, 2<<29, 1};
    
    for(int i = 0 ; i< map2.x; ++i){
        map2.data[i] = i;
    }
    /*
    auto map2 = new simpleMatrix<T>{data, 2<<29, 1};
    for(int i = 0 ; i< map2->x; ++i){
        map2->data[i] = i;
    }
    return map2;*/
    return std::move(map2);
}

inline double fastPrecisePow(double a, double b) {
  // calculate approximation with fraction of the exponent
  int e = (int) b;
  union {
    double d;
    int x[2];
  } u = { a };
  u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
  u.x[0] = 0;

  // exponentiation by squaring with the exponent's integer part
  // double r = u.d makes everything much slower, not sure why
  double r = 1.0;
  while (e) {
    if (e & 1) {
      r *= a;
    }
    a *= a;
    e >>= 1;
  }

  return r * u.d;
}


template<class T>
simple3DVector<T> equivalentPotentialTemperature_C(simple3DVector<T>&& t, simple3DVector<T>&& q, simple3DVector<T>&& sp){
    long int size = t.x*t.y*t.z;
    T* ept_data = new T[size];
    simple3DVector<T> ept{ept_data, t.x, t.y, t.z};

    const double cp = 1.00482;
    const double cw = 4.18674;
    const double Rd = 287.05;
    const double val = Rd/(cp*1000.0);

    for(int level = 0; level < t.x; ++level){
        for(int lat = 0; lat < t.y; ++lat){
            for(int lon = 0; lon < t.z; ++lon){
                int pos = lon+ lat*t.z + level*t.z*t.y;
                double t_ = t[pos]-273.15;
                double q_ = q[pos];
                double sp_ = sp[pos];
                double L = 2500.78-2.325734*t_;
                t_ += q_*(L/(cp+q_*cw)) + 273.15;
                t_ *= std::pow(100000/sp_, val);
                ept[pos] = t_;
            }
        }
    }
    return std::move(ept);
}

template<class T>
pybind11::array_t<T, CMASK_> calculateEquivalentPotentialTemperature(pybind11::array_t<T , CMASK_ >& temperature, pybind11::array_t<T , CMASK_ >& specific_humidity, pybind11::array_t<T , CMASK_ >& pressure){
    return vecToPy3DNoCopy(std::move(equivalentPotentialTemperature_C(pyToVec3D(temperature),pyToVec3D(specific_humidity),pyToVec3D(pressure))));
}


