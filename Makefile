all:
	cython roi_pooling_layer.pyx
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o roi_pooling_layer.so roi_pooling_layer.c
