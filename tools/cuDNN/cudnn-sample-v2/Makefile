CUDA_PATH=/usr/local/cuda
CUDNN_PATH=$(CUDA_PATH)/cudnn

CC=g++
CFLAGS=-I. -I$(CUDA_PATH)/include -I$(CUDNN_PATH) -IFreeImage/include -IUtilNPP
LIBS=-lcudart -lnppi -lnppc -lcublas -lcudnn -lfreeimage -lm -lstdc++
LFLAGS=-L$(CUDA_PATH)/lib64 -L$(CUDNN_PATH) -L./FreeImage/lib/linux/x86_64 $(LIBS)

OBJ = mnistCUDNN.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

mnistCUDNN: $(OBJ)
	gcc -o $@ $^ $(LFLAGS)

clean:
	rm *.o ./mnistCUDNN
