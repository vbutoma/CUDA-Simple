all:
	nvcc --std=c++11 -L/usr/local/cuda-8.0/lib64 /usr/local/cuda-8.0/lib64/libcudart_static.a `pkg-config --cflags opencv` `pkg-config --libs opencv` main.cu -o correction.run
