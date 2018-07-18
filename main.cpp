
#include <time.h>
#include <iostream>
#include <memory>
#include <x86intrin.h>
#include "gflags/gflags.h"
#include "paddle/fluid/inference/paddle_inference_api.h"
#include <string>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <iomanip>
#include <unistd.h>
#include <sys/types.h>

DEFINE_string(modeldir, "", "Directory of the inference model.");
DEFINE_int32(batch_size, 1, "Size of Batch of images to be processed");
DEFINE_int32(channels, 3, "Number of channels");
DEFINE_int32(height, 224, "Height of image");
DEFINE_int32(width, 224, "Width of Image");
DEFINE_int32(iterations, 1, "Number of Iterations (executions of Batches) to perform");
DEFINE_int32(fmaspc, 0,
    "Number of mul and add instructions that can be done within one cycle of CPU's core. Default(0) is guess value based on /proc/cpuinfo");

struct platform_info
{
    long num_logical_processors;
    long num_physical_processors_per_socket;
    long num_hw_threads_per_socket;
    unsigned int num_ht_threads; 
    unsigned int num_total_phys_cores;
    unsigned long long tsc;
    unsigned long long max_bandwidth; 
};

class nn_hardware_platform
{
    public:
        nn_hardware_platform() : m_num_logical_processors(0), m_num_physical_processors_per_socket(0), m_num_hw_threads_per_socket(0) ,m_num_ht_threads(1), m_num_total_phys_cores(1), m_tsc(0), m_fmaspc(0), m_max_bandwidth(0)
        {
#ifdef __linux__
            m_num_logical_processors = sysconf(_SC_NPROCESSORS_ONLN);
        
            m_num_physical_processors_per_socket = 0;

            std::ifstream ifs;
            ifs.open("/proc/cpuinfo"); 

            // If there is no /proc/cpuinfo fallback to default scheduler
            if(ifs.good() == false) {
                m_num_physical_processors_per_socket = m_num_logical_processors;
                assert(0);  // No cpuinfo? investigate that
                return;   
            }
            std::string cpuinfo_content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
            std::stringstream cpuinfo_stream(cpuinfo_content);
            std::string cpuinfo_line;
            std::string cpu_name;
            while(std::getline(cpuinfo_stream,cpuinfo_line,'\n')){
                if((m_num_physical_processors_per_socket == 0) && (cpuinfo_line.find("cpu cores") != std::string::npos)) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of physical cores per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find(":") + 1) ) >> m_num_physical_processors_per_socket; 
                }
                if(cpuinfo_line.find("siblings") != std::string::npos) {
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of HW threads per socket
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find(":") + 1) ) >> m_num_hw_threads_per_socket; 
                }

                if(cpuinfo_line.find("model") != std::string::npos) {
                    cpu_name = cpuinfo_line;
                    // convert std::string into number eg. skip colon and after it in the same line  should be number of HW threads per socket
                    float ghz_tsc = 0.0f;
                    std::stringstream( cpuinfo_line.substr(cpuinfo_line.find("@") + 1) ) >> ghz_tsc; 
                    m_tsc = static_cast<unsigned long long>(ghz_tsc*1000000000.0f);
                    
                    // Maximal bandwidth is Xeon 68GB/s , Brix 25.8GB/s
                    if(cpuinfo_line.find("Xeon") != std::string::npos) {
                      m_max_bandwidth = 68000;  //68 GB/s      -- XEONE5
                    } 
                    
                    if(cpuinfo_line.find("i7-4770R") != std::string::npos) {
                      m_max_bandwidth = 25800;  //25.68 GB/s      -- BRIX
                    } 
                }
                
                // determine instruction set (AVX, AVX2, AVX512)
                if(m_fmaspc == 0) {
                  if(FLAGS_fmaspc != 0) {
                    m_fmaspc = FLAGS_fmaspc;
                  } else {
                    if (cpuinfo_line.find(" avx") != std::string::npos) {
                      m_fmaspc = 8;   // On AVX instruction set we have one FMA unit , width of registers is 256bits, so we can do 8 muls and adds on floats per cycle
                      if (cpuinfo_line.find(" avx2") != std::string::npos) {
                        m_fmaspc = 16;   // With AVX2 instruction set we have two FMA unit , width of registers is 256bits, so we can do 16 muls and adds on floats per cycle
                      }
                      if (cpuinfo_line.find(" avx512") != std::string::npos) {
                        m_fmaspc = 32;   // With AVX512 instruction set we have two FMA unit , width of registers is 512bits, so we can do 32 muls and adds on floats per cycle
                      }

                    } 
                  }
                }
            }
            // If no FMA ops / cycle was given/found then raise a concern
            if(m_fmaspc == 0) {
              throw std::string("No AVX instruction set found. Please use \"--fmaspc\" to specify\n");
            }

            // There is cpuinfo, but parsing did not get quite right? Investigate it
            assert( m_num_physical_processors_per_socket > 0);
            assert( m_num_hw_threads_per_socket > 0);

            // Calculate how many threads can be run on single cpu core , in case of lack of hw info attributes assume 1
            m_num_ht_threads =  m_num_physical_processors_per_socket != 0 ? m_num_hw_threads_per_socket/ m_num_physical_processors_per_socket : 1;
            // calculate total number of physical cores eg. how many full Hw threads we can run in parallel
            m_num_total_phys_cores = m_num_hw_threads_per_socket != 0 ? m_num_logical_processors / m_num_hw_threads_per_socket * m_num_physical_processors_per_socket : 1;

            std::cout << "Platform:" << std::endl << "  " << cpu_name << std::endl 
                      << "  number of physical cores: " << m_num_total_phys_cores << std::endl; 

            ifs.close(); 

#endif
        }
    // Function computing percentage of theretical efficiency of HW capabilities
    float compute_theoretical_efficiency(unsigned long long start_time, unsigned long long end_time, unsigned long long num_fmas)
    {
      // Num theoretical operations
      // Time given is there
      return 100.0*num_fmas/((float)(m_num_total_phys_cores*m_fmaspc))/((float)(end_time - start_time));
    }

    void get_platform_info(platform_info& pi)
    {
       pi.num_logical_processors = m_num_logical_processors; 
       pi.num_physical_processors_per_socket = m_num_physical_processors_per_socket; 
       pi.num_hw_threads_per_socket = m_num_hw_threads_per_socket;
       pi.num_ht_threads = m_num_ht_threads;
       pi.num_total_phys_cores = m_num_total_phys_cores;
       pi.tsc = m_tsc;
       pi.max_bandwidth = m_max_bandwidth;
    }
    private:
        long m_num_logical_processors;
        long m_num_physical_processors_per_socket;
        long m_num_hw_threads_per_socket;
        unsigned int m_num_ht_threads;
        unsigned int m_num_total_phys_cores;
        unsigned long long m_tsc;
        short int m_fmaspc;
        unsigned long long m_max_bandwidth;
};


void fill_data(std::unique_ptr<float[]>& data, unsigned int count)
{
  for (unsigned int i = 0; i< count; ++i) {
    *(data.get() + i) = i;
  }
}


int main(int argc, char** argv) {
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_modeldir.empty()) {
    // Example:
    std::cout << "Error: Directory with Model not specified. Example of running: ./test_paddle_fluid --modeldir=path/to/your/model" << std::endl;
    exit(1);
  }

  nn_hardware_platform machine;
  platform_info pi;
  machine.get_platform_info(pi);

  paddle::NativeConfig config;
  config.param_file = FLAGS_modeldir + "/resent50-params";
  config.prog_file = FLAGS_modeldir + "/__model__";
  config.use_gpu = false;
  config.device = 0;

  auto predictor =
      paddle::CreatePaddlePredictor<paddle::NativeConfig, paddle::PaddleEngineKind::kNative>(config);


  std::vector<int> shape;
  shape.push_back(FLAGS_batch_size);
  shape.push_back(FLAGS_channels);
  shape.push_back(FLAGS_height);
  shape.push_back(FLAGS_width);

  auto count = [](std::vector<int>& shapevec)
  {
    auto sum = shapevec.size() > 0 ? 1 : 0;
    for (unsigned int i=0; i < shapevec.size(); ++i) {
      sum *= shapevec[i];
    }
    return sum;
  }; 
  
  std::unique_ptr<float[]> data(new float[count(shape)]);
  fill_data(data, count(shape));

  // Inference.
  paddle::PaddleTensor input{
      .name = "xx",
      .shape = shape,
      .data = paddle::PaddleBuf(data.get(), count(shape)*sizeof(float)),
      .dtype = paddle::PaddleDType::FLOAT32};

  std::vector<paddle::PaddleTensor> output;

  auto t1 = __rdtsc();
  for (int i =0; i<FLAGS_iterations; ++i) {
    predictor->Run({input}, &output);
  }
  auto t2 = __rdtsc();

  std::cout << std::endl << "---> " << "Inference" << " on average takes " << (t2 -t1)*1000.0f/((float)pi.tsc*FLAGS_iterations) << " ms" << " Throughput: " << shape[0]/((t2 -t1)/((float)pi.tsc*FLAGS_iterations)) << " Images/sec";

  std::cout << std::endl;


  auto& tensor = output.front();

  return 0;
}
