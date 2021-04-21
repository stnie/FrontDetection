#pragma once
#include <pybind11/numpy.h>
#include <vector>


#define CMASK_ pybind11::array::c_style | pybind11::array::forcecast


template<class T>
struct simpleNDVector{
    T* data;
    long int dimensions;
    long int dims[];

    void free_data(){
        delete[] data;
    }
};


template<class T>
struct simple3DVector{
    T* data;
    long int x;
    long int y;
    long int z;

    T& operator[](long int pos){
        return data[pos];
    }

    void free_data(){
        delete[] data;
    }
};

template<class T>
struct simpleMatrix{
    T* data;
    long int x;
    long int y;

    void free_data(){
        delete[] data;
    }
};

template<class T>
struct simpleVector{
    T* data;
    long int x;

    void free_data(){
        delete[] data;
    }
};

template<class T>
simpleVector<T> pyToVec1D(pybind11::array_t<T, CMASK_>& pyarray){
    // check dims
    if(pyarray.ndim() != 1) throw pybind11::cast_error();
    auto pyarray_unchecked = pyarray.template mutable_unchecked<1>();
    return std::move(simpleVector<T>{&pyarray_unchecked(0), pyarray_unchecked.shape(0)});
}

template<class T>
simpleMatrix<T> pyToVec2D(pybind11::array_t<T, CMASK_>& pyarray){
    // check dims
    if(pyarray.ndim() != 2) throw pybind11::cast_error();
    std::vector<T*> cvec(pyarray.shape(0));
    auto pyarray_unchecked = pyarray.template mutable_unchecked<2>();
    return std::move(simpleMatrix<T>{&pyarray_unchecked(0,0), pyarray.shape(0), pyarray.shape(1)});
}

template<class T>
simple3DVector<T> pyToVec3D(pybind11::array_t<T, CMASK_>& pyarray){
    // check dims
    if(pyarray.ndim() != 3) throw pybind11::cast_error();
    std::vector<T*> cvec(pyarray.shape(0));
    auto pyarray_unchecked = pyarray.template mutable_unchecked<3>();
    return std::move(simple3DVector<T>{&pyarray_unchecked(0,0,0), pyarray.shape(0), pyarray.shape(1), pyarray.shape(2)});
}


template<class T>
pybind11::array_t<T, CMASK_> vecToPy2D(simpleMatrix<T>&& mat){
    return pybind11::array_t<T, CMASK_>(
        pybind11::buffer_info(
            mat.data,                               /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            pybind11::format_descriptor<T>::format(), /* Python struct-style format descriptor */
            2,                                      /* Number of dimensions */
            std::vector<long int>{ mat.x, mat.y },                 /* Buffer dimensions */
            std::vector<size_t>{ sizeof(T) * mat.y,             /* Strides (in bytes) for each index */
              sizeof(T) }
        )
    );
}

template<class T>
pybind11::array_t<T, CMASK_> vecToPy3D(simple3DVector<T>&& vec){
    return pybind11::array_t<T, CMASK_>(
        pybind11::buffer_info(
            vec.data,                               /* Pointer to buffer */
            sizeof(T),                          /* Size of one scalar */
            pybind11::format_descriptor<T>::format(), /* Python struct-style format descriptor */
            3,                                      /* Number of dimensions */
            std::vector<long int>{ vec.x, vec.y, vec.z },                 /* Buffer dimensions */
            std::vector<size_t>{ sizeof(T) * vec.y * vec.z, sizeof(T) * vec.y,             /* Strides (in bytes) for each index */
              sizeof(T) }
        )
    );
}


template<class T>
pybind11::array_t<T, CMASK_> vecToPy2DNoCopy(simpleMatrix<T>&& mat){
    auto capsule = pybind11::capsule(mat.data, [](void* matdata) {delete[] reinterpret_cast<T*>(matdata);});
    return pybind11::array_t<T, CMASK_>(std::vector<long int>{mat.x, mat.y}, mat.data, capsule);
}

template<class T>
pybind11::array_t<T, CMASK_> vecToPy3DNoCopy(simple3DVector<T>&& vec){
    auto capsule = pybind11::capsule(vec.data, [](void* vecdata) {delete[] reinterpret_cast<T*>(vecdata);});
    return pybind11::array_t<T, CMASK_>(std::vector<long int>{vec.x, vec.y, vec.z}, vec.data, capsule);
}