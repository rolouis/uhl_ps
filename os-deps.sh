# CLI Tools
sudo apt install -y imagemagick libjxr-tools heif-gdk-pixbuf
sudo apt install -y libheif* 
cd openjpeg
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make