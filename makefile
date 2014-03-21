SHELL=/bin/bash

#On doc machines, do export PATH=$PATH:/usr/local/cuda/bin

CPPFLAGS += -std=c++11 -W -Wall -g -MMD
CPPFLAGS += -O3
CPPFLAGS += -I include -I src

LDLIBS = -lm
# -lOpenCL

# For your makefile, add TBB and OpenCL as appropriate
# OpenCL stuff
#OpenCL_DIR = opencl_sdk
#OpenCL_INC_DIR = $(OpenCL_DIR)/include
#OpenCL_LIB_DIR = $(OpenCL_DIR)/lib/cygwin/x86_64

#CPPFLAGS += -I $(OpenCL_INC_DIR)
#LDFLAGS += -L $(OpenCL_LIB_DIR)

# Cuda on DoC machines
Cuda_DIR = /usr/local/cuda
Cuda_INC_DIR = $(Cuda_DIR)/include
Cuda_LIB_DIR = $(Cuda_DIR)/lib64

CPPFLAGS += -I $(Cuda_INC_DIR)
LDFLAGS += -L $(Cuda_LIB_DIR)
LDLIBS += -lcudart

# Launch client and server connected by pipes
launch_pipes : src/bitecoin_server src/bitecoin_client
	-rm .fifo_rev
	mkfifo .fifo_rev
	# One direction via pipe, other via fifo
	src/bitecoin_client Clockwork 3 file .fifo_rev - | (src/bitecoin_server server1 3 file - .fifo_rev &> /dev/null)

# Launch an "infinite" server, that will always relaunch
launch_infinite_server : src/bitecoin_server
	while [ 1 ]; do \
		src/bitecoin_server server1-$USER 3 tcp-server 4000; \
	done;

# Launch a client connected to a local server
connect_local : src/bitecoin_client
	src/bitecoin_client Clockwork 3 tcp-client localhost 4000


EXCHANGE_ADDR = 155.198.117.237
EXCHANGE_PORT = 4123
	
# Launch a client connected to a shared exchange
connect_exchange : src/bitecoin_client
	src/bitecoin_client Clockwork 3 tcp-client $(EXCHANGE_ADDR)  $(EXCHANGE_PORT)


#src/bitecoin_client: src/bitecoin_client.o


#VS options -gencode=arch=compute_30,code=\"sm_30,compute_30\" --use-local-env --cl-version 2012 -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin\x86_amd64"  -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\include" --opencc-options -LIST:source=on -G -lineinfo  --keep-dir x64\Release -maxrregcount=0 --ptxas-options=-v --machine 64 --compile -cudart static     -DWIN64 -DNDEBUG -D_CONSOLE -D_MBCS -Xcompiler "/EHsc /W3 /nologo /Ox /Zi  /MD  " -o x64\Release\Clockwork_kernels.cu.obj "C:\Users\Oskar\Documents\hpce-2013-cw6\src\Clockwork_kernels.cu" 

src/Clockwork_kernels.o: src/Clockwork_kernels.cu
	nvcc -O2 -gencode=arch=compute_30,code=\"sm_30,compute_30\" --ptxas-options=-v --machine 64 -cudart static -I Cuda_INC_DIR -c $< -o $@

#src/Clockwork_kernels: src/Clockwork_kernels.o
#	nvcc $^ -o $@

#test_cuda: src/Clockwork_kernels

.PHONY: clean
clean: 
	-rm src/bitecoin_client.exe #src/bitecoin_server.exe
#-include src/bitecoin_client.d