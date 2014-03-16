#ifndef  Clockwork_hpp
#define  Clockwork_hpp

#ifdef _MSC_VER
#include <malloc.h>
#else
#include <alloca.h>
#endif

#define __CL_ENABLE_EXCEPTIONS 
#include "CL/cl.hpp"

namespace bitecoin{

class Clockwork
{
private:

public:
	Clockwork(){
		try{

			std::vector<cl::Platform> platforms;
			
			cl::Platform::get(&platforms);
			if(platforms.size()==0)
				throw std::runtime_error("No OpenCL platforms found.");
			
			std::cerr<<"Found "<<platforms.size()<<" platforms\n";
			for(unsigned i=0;i<platforms.size();i++){
				std::string vendor=platforms[0].getInfo<CL_PLATFORM_VENDOR>();
				std::cerr<<"  Platform "<<i<<" : "<<vendor<<"\n";
			}
			
			int selectedPlatform=0;
			if(getenv("HPCE_SELECT_PLATFORM")){
				selectedPlatform=atoi(getenv("HPCE_SELECT_PLATFORM"));
			}
			std::cerr<<"Choosing platform "<<selectedPlatform<<"\n";
			cl::Platform platform=platforms.at(selectedPlatform);
			
			std::vector<cl::Device> devices;
			platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);	
			if(devices.size()==0){
				throw std::runtime_error("No opencl devices found.\n");
			}
				
			std::cerr<<"Found "<<devices.size()<<" devices\n";
			for(unsigned i=0;i<devices.size();i++){
				std::string name=devices[i].getInfo<CL_DEVICE_NAME>();
				std::cerr<<"  Device "<<i<<" : "<<name<<"\n";
			}
			
			int selectedDevice=0;
			if(getenv("HPCE_SELECT_DEVICE")){
				selectedDevice=atoi(getenv("HPCE_SELECT_DEVICE"));
			}
			std::cerr<<"Choosing device "<<selectedDevice<<"\n";
			cl::Device device=devices.at(selectedDevice);
			
			cl::Context context(devices);

		}catch(const std::exception &e){
			std::cerr<<"Clockwork.hpp Exception : "<<e.what()<<std::endl;
			throw e;
		}
	}

	~Clockwork();
};

}

#endif //Clockwork_hpp