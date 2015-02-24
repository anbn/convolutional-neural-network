#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "network.hpp"
#include "image.hpp"

using namespace my_nn;

void print_vec(std::string s, vec_t v) {
    std::cout<<s;
    int i=0;
    for (auto e : v) {
        std::cout<<(i++)<<" "<<e<<"   ";
    }
    std::cout<<"\n";
}


int
main(int argc, const char *argv[])
{
    vec_t input {0.0, 1.0, 1.0, 0.0};
    vec_t soll {1,0,1,0};
    vec_t output;

    convolutional_layer<sigmoid> C1(32,28,2);
    subsampling_layer<sigmoid> S2(28,14,2);
    convolutional_layer<sigmoid> C3(14,12,5);
    subsampling_layer<sigmoid> S4(12,6,5);
    convolutional_layer<sigmoid> C5(6,1,5);
    output_layer<sigmoid> O6(5,1);
    
    vec_t test(32*32);
    for(int i=0; i<32; i++){
        test[32*i+i] = 1;
        test[(32*i+i+1)%(32*32)] = 1;
        test[32*i+16] = 0.2;
        test[32*20+i] = 0.5;
    }
    Image<my_nn::float_t> img1(32,32,std::begin(test),std::end(test));
    cv::imshow("C1 input", img1.toIntensity().exportMat());

    C1.feed_forward(1 /*in_fm*/, 32, test);

    Image<my_nn::float_t> img(28, 28*2, std::begin(C1.output()), std::end(C1.output()));
    cv::imshow("C1 output", img.toIntensity(0,1).exportMat());


    while(cv::waitKey(0)!=27);
    std::cout<<"Hallo Welt\n";
    return 0;
}

