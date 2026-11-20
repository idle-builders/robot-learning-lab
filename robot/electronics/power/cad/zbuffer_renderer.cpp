// Orthographic triangle-depth renderer for evidence images. No geometry changes.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <vector>
#include <limits>
int main(int argc,char** argv){
 if(argc!=3)return 2;
 std::ifstream f(argv[1],std::ios::binary);uint32_t W,H,N;
 f.read((char*)&W,4);f.read((char*)&H,4);f.read((char*)&N,4);
 if(!f||!W||!H)return 3;
 std::vector<float> depth((size_t)W*H,-std::numeric_limits<float>::infinity());
 std::vector<uint8_t> rgb((size_t)W*H*3);
 for(size_t i=0;i<(size_t)W*H;i++){rgb[i*3]=246;rgb[i*3+1]=245;rgb[i*3+2]=241;}
 for(uint32_t n=0;n<N;n++){
  float t[12];f.read((char*)t,48);if(!f)return 4;
  double x0=t[0],y0=t[1],x1=t[3],y1=t[4],x2=t[6],y2=t[7];
  double den=(y1-y2)*(x0-x2)+(x2-x1)*(y0-y2);if(std::abs(den)<1e-8)continue;
  int xmin=std::max(0,(int)std::floor(std::min({x0,x1,x2}))),xmax=std::min((int)W-1,(int)std::ceil(std::max({x0,x1,x2})));
  int ymin=std::max(0,(int)std::floor(std::min({y0,y1,y2}))),ymax=std::min((int)H-1,(int)std::ceil(std::max({y0,y1,y2})));
  for(int y=ymin;y<=ymax;y++)for(int x=xmin;x<=xmax;x++){
   double px=x+.5,py=y+.5;
   double a=((y1-y2)*(px-x2)+(x2-x1)*(py-y2))/den;
   double b=((y2-y0)*(px-x2)+(x0-x2)*(py-y2))/den;double c=1-a-b;
   if(a<0||b<0||c<0)continue;
   float z=a*t[2]+b*t[5]+c*t[8];size_t k=(size_t)y*W+x;
   if(z>depth[k]){depth[k]=z;for(int j=0;j<3;j++)rgb[k*3+j]=(uint8_t)std::clamp(t[9+j]*255.f,0.f,255.f);}
  }
 }
 std::ofstream o(argv[2],std::ios::binary);o<<"P6\n"<<W<<" "<<H<<"\n255\n";o.write((char*)rgb.data(),rgb.size());return o?0:5;
}
