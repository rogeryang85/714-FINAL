# Running the Code

## Running Needle with the original CUDA code

1. Navigate to `src/ndarray_backend_cuda.cc` and set the `efficient` flag
   in `matmul` function to false
2. Run `make`

## Test optimized matmul kernel (eff)

```
cd tests
python test_eff.py --begin 0 --num 1
```

## Test cuBLAS matmul kernel with TVM
Firstly, install TVM.
```

# make sure to start with a fresh environment

conda env remove -n tvm-build-venv

# create the conda environment with build dependency

conda create -n tvm-build-venv -c conda-forge \
 "llvmdev>=15" \
 "cmake>=3.24" \
 git \
 python=3.11

# enter the build environment

conda activate tvm-build-venv
```
Then,
```
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
rm -rf build && mkdir build && cd build

# Specify the build configuration via CMake options

cp ../cmake/config.cmake .

# controls default compilation flags (Candidates: Release, Debug, RelWithDebInfo)

echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake

# LLVM is a must dependency for compiler end

echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake

# GPU CUDA turn on if needed

echo "set(USE_CUDA ON)" >> config.cmake

# cuBLAS support, turn on if needed

echo "set(USE_CUBLAS ON)" >> config.cmake

cmake .. && cmake --build . --parallel $(nproc)
export TVM_HOME=/path-to-tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
```
Then `cd tests` and run `python3 test_cublas.py`.
