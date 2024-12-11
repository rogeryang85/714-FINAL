# Running the Code 
## Running Needle with the original CUDA code 
  1. Navigate to ```src/ndarray_backend_cuda.cc``` and set the ```efficient``` flag 
  in ```matmul``` function to false
  2. Run ```make```
## Running Needle with optimized matmul kernel (eff) 
  1. Navigate to ```src/ndarray_backend_cuda.cc``` and set the ```efficient``` flag 
  in ```matmul``` function to true
  2. Run ```make```
