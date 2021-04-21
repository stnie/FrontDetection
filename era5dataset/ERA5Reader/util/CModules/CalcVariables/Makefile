
all: example.so

example.so: pybind11_wrapper.cpp
	g++ -O3 -Wall -shared -std=c++11 -I ../PyToC/ -fPIC -I/usr/include/python3.8 -I/home/stefan/Documents/Binary/Front-Detection/FrontDetectionVenv38/lib/python3.8/site-packages/pybind11/include pybind11_wrapper.cpp -o example.so
