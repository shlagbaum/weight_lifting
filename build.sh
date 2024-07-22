git clone https://gitlab.com/libeigen/eigen.git
mkdir build
cd build
cmake ../
sudo make install

git clone https://github.com/gadomski/fgt
cmake -D CMAKE_CXX_STANDARD=14 -D EIGEN3_INCLUDE_DIR=/usr/local/include/eigen3 -D BUILD_SHARED_LIBS=1 -D CMAKE_BUILD_TYPE=Release ../
make test

