#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define GRADIENT_CHECK  0
#define VERBOSE         0

#include "network.hpp"
#include "image.hpp"
#include "mnist_reader.hpp"
#include "orl_reader.hpp"

using namespace nn;

void print_vec(std::string s, vec_t v) {
    std::cout<<s;
    for (auto e : v) {
        std::cout<<" "<<e;
    }
    std::cout<<"\n";
}


#if GRADIENT_CHECK
void
gc_fullyconnected()
{
    const nn::float_t gc_epsilon = 0.00001;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(16);

    neural_network<sigmoid> nn;
    fullyconnected_layer<sigmoid> L1(4,8);
    fullyconnected_layer<sigmoid> L2(8,6);
    fullyconnected_layer<sigmoid> L3(6,4);
    nn.add_layer(&L1);
    nn.add_layer(&L2);
    nn.add_layer(&L3);

    vec_t input(4), soll(4);
    randomize(input.begin(), input.end(), 0, 1);
    for (auto& v : input) {
        v = v<0.5 ? 0 : 1;
    }
    soll = input;
    std::reverse(soll.begin(), soll.end());
    
    std::cout<<"- L3 ---------------------------\n";
    nn.gc_check_weights(input, soll, &L3);
    std::cout<<"- L3 bias ----------------------\n";
    nn.gc_check_bias(input, soll, &L3);
    std::cout<<"- L2 ---------------------------\n";
    nn.gc_check_weights(input, soll, &L2);
    std::cout<<"- L2 bias-----------------------\n";
    nn.gc_check_bias(input, soll, &L2);
}


void
gc_cnn_training()
{
    mnist_reader mnist;
    mnist.read( "data/mnist/train-images-idx3-ubyte",
                "data/mnist/train-labels-idx1-ubyte", 10);

    const nn::float_t gc_epsilon = 0.00001;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(16);

    neural_network<tan_h> nn;
    convolutional_layer<tan_h>  C1(28 /* in_width*/, 24 /* out_width*/,  1 /*in_fm*/,  6 /*out_fm*/);
    subsampling_layer<tan_h>    S2(24 /* in_width*/, 12 /* out_width*/,  6 /*fm*/,     2 /*block_size*/);
    convolutional_layer<tan_h>  C3(12 /* in_width*/, 10 /* out_width*/,  6 /*in_fm*/, 16 /*out_fm*/);
#define _ false
#define X true
    const bool connection[] = {
        X, _, _, _, X, X, X, _, _, X, X, X, X, _, X, X,
        X, X, _, _, _, X, X, X, _, _, X, X, X, X, _, X,
        X, X, X, _, _, _, X, X, X, _, _, X, _, X, X, X,
        _, X, X, X, _, _, X, X, X, X, _, _, X, _, X, X,
        _, _, X, X, X, _, _, X, X, X, X, _, X, X, _, X,
        _, _, _, X, X, X, _, _, X, X, X, X, _, X, X, X
    };
#undef _
#undef X
    C3.set_connection(connection, 6*16);
    subsampling_layer<tan_h>    S4(10 /* in_width*/,  5 /* out_width*/, 16 /*fm*/,     2 /*block_size*/);
    convolutional_layer<tan_h>  C5( 5 /* in_width*/,  1 /* out_width*/, 16 /*in_fm*/, 12 /*out_fm*/);
    fullyconnected_layer<tan_h> O6(12 /* in_width*/, 14 /* out_width*/);
    fullyconnected_layer<tan_h> O7(14 /* in_width*/, 10 /* out_width*/);
    nn.add_layer(&C1);
    nn.add_layer(&S2);
    nn.add_layer(&C3);
    nn.add_layer(&S4);
    nn.add_layer(&C5);
    nn.add_layer(&O6);
    nn.add_layer(&O7);


    vec_t soll {-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8};
    int last_label = 0;

    int num_example = 2 % mnist.num_examples();
    nn.forward(mnist.image(num_example).data());

    soll[last_label] = -0.8;
    last_label = mnist.label(num_example);
    soll[last_label] = 0.8;


    std::cout<<"- O7 ---------------------------\n";
    nn.gc_check_weights(mnist.image(num_example).data(), soll, &O7);
    std::cout<<"- O7 bias ----------------------\n";
    nn.gc_check_bias(mnist.image(num_example).data(), soll, &O7);

    std::cout<<"- O6 ---------------------------\n";
    nn.gc_check_weights(mnist.image(num_example).data(), soll, &O6);
    std::cout<<"- O6 bias ----------------------\n";
    nn.gc_check_bias(mnist.image(num_example).data(), soll, &O6);

    std::cout<<"- C5 ---------------------------\n";
    nn.gc_check_weights(mnist.image(num_example).data(), soll, &C5);
    std::cout<<"- C5 bias ----------------------\n";
    nn.gc_check_bias(mnist.image(num_example).data(), soll, &C5);

    std::cout<<"- S4 ---------------------------\n";
    nn.gc_check_weights(mnist.image(num_example).data(), soll, &S4);
    std::cout<<"- S4 bias ----------------------\n";
    nn.gc_check_bias(mnist.image(num_example).data(), soll, &S4);

    std::cout<<"- C3 ---------------------------\n";
    nn.gc_check_weights(mnist.image(num_example).data(), soll, &C3);
    std::cout<<"- C3 bias ----------------------\n";
    nn.gc_check_bias(mnist.image(num_example).data(), soll, &C3);

    std::cout<<"- S2 ---------------------------\n";
    nn.gc_check_weights(mnist.image(num_example).data(), soll, &S2);
    std::cout<<"- S2 bias ----------------------\n";
    nn.gc_check_bias(mnist.image(num_example).data(), soll, &S2);

    std::cout<<"- C1 ---------------------------\n";
    nn.gc_check_weights(mnist.image(num_example).data(), soll, &C1);
    std::cout<<"- C1 bias ----------------------\n";
    nn.gc_check_bias(mnist.image(num_example).data(), soll, &C1);
}
#endif

void
fullyconnected_test()
{
    fullyconnected_layer<tan_h> L1(4,8);
    fullyconnected_layer<tan_h> L2(8,6);
    fullyconnected_layer<tan_h> L3(6,5);
    fullyconnected_layer<tan_h> L4(5,4);

    neural_network<tan_h> nn;
    nn.add_layer(&L1);
    nn.add_layer(&L2);
    nn.add_layer(&L3);
    nn.add_layer(&L4);
    
    vec_t input(4);
    vec_t soll(4);

    nn::float_t moving_error = 1.0;
    for(int s=0; s<1000000; s++) {

        randomize(input.begin(), input.end(), 0, 1);
        for (auto& v : input) {
            v = v<0.5 ? -1 : 1;
        }
        soll = input;
        std::reverse(soll.begin(), soll.end());

        nn.forward(input);

        float error = 0.0;
        for (int i=0; i<soll.size(); i++) {
            error += (soll[i]-nn.output()[i])*(soll[i]-nn.output()[i]);
        }
        
        if(moving_error<0.15) {
            print_vec("INPUT: ", input);
            print_vec("SOLL:  ", soll);
            print_vec("IST:   ", nn.output());
        }
        std::cout<< s << "   error: "<<error<<" \t("<<moving_error<<")\n";

        nn.backward(input, soll);

        moving_error = 0.9*moving_error + 0.1*error;
        if(moving_error<0.1) exit(0);
    }
}


void
cnn_training_test()
{
    const int steps = 100000;
    mnist_reader mnist;
    mnist.read("data/mnist/", std::min(steps, 0));

    neural_network<tan_h> nn;
    convolutional_layer<tan_h>  C1(28 /* in_width*/, 24 /* out_width*/, 1 /*in_fm*/,   6 /*out_fm*/);
    subsampling_layer<tan_h>    S2(24 /* in_width*/, 12 /* out_width*/, 6 /*fm*/,      2 /*block_size*/);
    convolutional_layer<tan_h>  C3(12 /* in_width*/, 10 /* out_width*/, 6 /*in_fm*/,  16 /*out_fm*/);
#define _ false
#define X true
    const bool connection[] = {
        X, _, _, _, X, X, X, _, _, X, X, X, X, _, X, X,
        X, X, _, _, _, X, X, X, _, _, X, X, X, X, _, X,
        X, X, X, _, _, _, X, X, X, _, _, X, _, X, X, X,
        _, X, X, X, _, _, X, X, X, X, _, _, X, _, X, X,
        _, _, X, X, X, _, _, X, X, X, X, _, X, X, _, X,
        _, _, _, X, X, X, _, _, X, X, X, X, _, X, X, X
    };
#undef _
#undef X
    C3.set_connection(connection, 6*16);
    subsampling_layer<tan_h>    S4(10 /* in_width*/,  5 /* out_width*/, 16 /*fm*/,     2 /*block_size*/);
    convolutional_layer<tan_h>  C5( 5 /* in_width*/,  1 /* out_width*/, 16 /*in_fm*/, 64 /*out_fm*/);
    fullyconnected_layer<tan_h> O6(64 /* in_width*/, 10 /* out_width*/);
    nn.add_layer(&C1);
    nn.add_layer(&S2);
    nn.add_layer(&C3);
    nn.add_layer(&S4);
    nn.add_layer(&C5);
    nn.add_layer(&O6);
    
    nn.set_learningrate(1.0/(10000));
    
    vec_t soll {-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8};
    int last_label = 0;
    for(int s=0; s<steps; s++) {
        
        //if (s!=0 && s%1000==0) nn.set_learningrate(1.0/(100000));
        
        int num_example = s % mnist.num_examples();

        nn.forward(mnist.image(num_example).data());
        
        soll[last_label] = -0.8;
        last_label = mnist.label(num_example);
        soll[last_label] =  0.8;

        nn::float_t error = nn.squared_error(soll);
        std::cout<<"Step: "<<s<<"\n";
        for (int o=0; o<soll.size(); o++)
            std::cout<<"    ["<<o<<"]: "<<soll[o]<<" vs "<<nn.output()[o]<<"\n";
        std::cout<<"    error "<<error<<"\n";

        nn.backward(mnist.image(num_example).data(), soll);

        if ( s%30000 < 50 ) {

            Image<nn::float_t> img_in(28, 1*28, std::begin(mnist.image(num_example).data()), std::end(mnist.image(num_example).data()));
            Image<nn::float_t> img_c1(24, 6*24, std::begin(C1.output()), std::end(C1.output()));
            Image<nn::float_t> img_s2(12, 6*12, std::begin(S2.output()), std::end(S2.output()));
            Image<nn::float_t> img_c3(10,16*10, std::begin(C3.output()), std::end(C3.output()));
            Image<nn::float_t> img_s4( 5, 16*5, std::begin(S4.output()), std::end(S4.output()));
            Image<nn::float_t> img_c5( 1,   64, std::begin(C5.output()), std::end(C5.output()));
            Image<nn::float_t> img_o6( 1,   10, std::begin(O6.output()), std::end(O6.output()));
            Image<nn::float_t> img_so( 1,   10, std::begin(soll), std::end(soll));

            cv::Mat img(cv::Size(400,200), CV_8UC1, 100);
            img_in.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(  0,0,img_in.width(), img_in.height())));
            img_c1.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect( 50,0,img_c1.width(), img_c1.height())));
            img_s2.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(100,0,img_s2.width(), img_s2.height())));
            img_c3.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(150,0,img_c3.width(), img_c3.height())));
            img_s4.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(200,0,img_s4.width(), img_s4.height())));
            img_c5.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(250,0,img_c5.width(), img_c5.height())));
            img_o6.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(300,0,img_o6.width(), img_o6.height())));
            img_so.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(301,0,img_o6.width(), img_o6.height())));

            cv::imshow("cnn", img);
            
            while(cv::waitKey(0)!=27);
        }
    }
}


int
main(int argc, const char *argv[])
{

#if GRADIENT_CHECK
    //gc_fullyconnected();
    //gc_cnn_training();
#endif

    //fullyconnected_test();
    //cnn_training_test();

    orl_reader_test();
    //mnist_reader_test();
    
    return 0;
}
