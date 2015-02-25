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

    convolutional_layer<sigmoid> C1(32,28,1,2);
    subsampling_layer<sigmoid> S2(28,14,2);
    convolutional_layer<sigmoid> C3(14,12,2,5);
    subsampling_layer<sigmoid> S4(12,6,5);
    convolutional_layer<sigmoid> C5(6,1,5,5);
    output_layer<sigmoid> O6(5,2);
    
    vec_t test(32*32);
    for(int i=0; i<32; i++){
        if(i>10) {
            test[32*i+i-1] = 0.5;
            test[32*i+i]   = 0.8;
            test[32*i+i+1] = 1.0;
        }
        test[32*20+i] = 1.0;
        test[32*i+20] = 1.0;
    }
    test[32*5+5] = 1.0;
    
    Image<my_nn::float_t> img_in(32,32,std::begin(test),std::end(test));
    //cv::imshow("C1 input", img_in.toIntensity().exportMat());

    C1.feed_forward(1 /*in_fm*/, 32 /*in_width*/, test);
    S2.feed_forward(2 /*in_fm*/, 28 /*in_width*/, C1.output());
    C3.feed_forward(2 /*in_fm*/, 14 /*in_width*/, S2.output());
    S4.feed_forward(5 /*in_fm*/, 12 /*in_width*/, C3.output());
    C5.feed_forward(5 /*in_fm*/,  6 /*in_width*/, S4.output());
    O6.feed_forward(5 /*in_dim*/, C5.output());


    cv::Mat img(cv::Size(400,100), CV_8UC1, 100);

    //print_vec("TEST: ", C1.output());

    Image<my_nn::float_t> img_c1(28, 28*2, std::begin(C1.output()), std::end(C1.output()));
    Image<my_nn::float_t> img_s2(14, 14*2, std::begin(S2.output()), std::end(S2.output()));
    Image<my_nn::float_t> img_c3(12, 12*5, std::begin(C3.output()), std::end(C3.output()));
    Image<my_nn::float_t> img_s4( 6,  6*5, std::begin(S4.output()), std::end(S4.output()));
    Image<my_nn::float_t> img_c5( 1,  1*5, std::begin(C5.output()), std::end(C5.output()));
    Image<my_nn::float_t> img_o6( 1,    2, std::begin(O6.output()), std::end(O6.output()));
    
    img_in.toIntensity( 0,1).exportMat().copyTo(img(cv::Rect(  0,0,img_in.width(), img_in.height())));
    img_c1.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect( 50,0,img_c1.width(), img_c1.height())));
    img_s2.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(100,0,img_s2.width(), img_s2.height())));
    img_c3.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(150,0,img_c3.width(), img_c3.height())));
    img_s4.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(200,0,img_s4.width(), img_s4.height())));
    img_c5.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(250,0,img_c5.width(), img_c5.height())));
    img_o6.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(300,0,img_o6.width(), img_o6.height())));

    cv::imshow("cnn", img);
    while(cv::waitKey(0)!=27);
    return 0;
}

