SHELL=/bin/bash

#On doc machines, do export PATH=$PATH:/usr/local/cuda/bin

CPPFLAGS += -std=c++11 -W -Wall -g -MMD
CPPFLAGS += -O3
CPPFLAGS += -I include -I src

LDLIBS = -lm -lOpenCL

# For your makefile, add TBB and OpenCL as appropriate
# OpenCL stuff
OpenCL_DIR = opencl_sdk
OpenCL_INC_DIR = $(OpenCL_DIR)/include
OpenCL_LIB_DIR = $(OpenCL_DIR)/lib/cygwin/x86_64

CPPFLAGS += -I $(OpenCL_INC_DIR)
LDFLAGS += -L $(OpenCL_LIB_DIR)

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


src/Clockwork_kernels.o: src/Clockwork_kernels.cu
	nvcc -c $< -o $@

src/Clockwork_kernels: src/Clockwork_kernels.o
	nvcc $^ -o $@

test_cuda: src/Clockwork_kernels

.PHONY: clean
clean: 
	-rm src/bitecoin_client.exe #src/bitecoin_server.exe
#-include src/bitecoin_client.d