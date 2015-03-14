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
    for (auto e : v) {
        std::cout<<" "<<e;
    }
    std::cout<<"\n";
}


void
fullyconnected_test()
{
    fullyconnected_layer<tan_h> L1(4,8);
    fullyconnected_layer<tan_h> L2(8,6);
    fullyconnected_layer<tan_h> L3(6,4);

    vec_t input(4);
    vec_t soll(4);

    my_nn::float_t moving_error = 1.0;
    for(int s=0; s<100000; s++) {

        randomize(input.begin(), input.end(), 0, 1);
        for (auto& v : input) {
            v = v<0.5 ? -1 : 1;
        }
        soll = input;
        std::reverse(soll.begin(), soll.end());

        L1.forward(input);
        L2.forward(L1.output());
        L3.forward(L2.output());

        float error = 0.0;
        for (int i=0; i<soll.size(); i++) {
            error += (soll[i]-L3.output()[i])*(soll[i]-L3.output()[i]);
        }
        
        if(moving_error<0.015) {
            print_vec("INPUT: ", input);
            print_vec("SOLL:  ", soll);
            print_vec("IST:   ", L3.output());
        }
        std::cout<< s << "   error: "<<error<<"\n";

        L3.backward(L2,soll);
        L2.backward(L1.output(), L3);
        L1.backward(input, L2);

        moving_error = 0.9*moving_error + 0.1*error;
        if(moving_error<0.01) exit(0);
    }

}

#if GRADIENT_CHECK
void
gc_fullyconnected()
{
    const my_nn::float_t gc_epsilon = 0.0001;
    std::cout.precision(32);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);

    fullyconnected_layer<sigmoid> L1(4,8);
    fullyconnected_layer<sigmoid> L2(8,6);
    fullyconnected_layer<sigmoid> L3(6,4);

    vec_t input(4), soll(4);
    randomize(input.begin(), input.end(), 0, 1);
    for (auto& v : input) {
        v = v<0.5 ? 0 : 1;
    }
    soll = input;
    std::reverse(soll.begin(), soll.end());
    
    for (int o=0; o<L3.out_dim(); o++) {
        for (int i=0; i<L3.in_dim(); i++) {

            my_nn::float_t weight = L3.get_weight(o*L3.in_dim() + i);

            L3.set_weight(o*L3.in_dim() + i, weight+gc_epsilon);
            L1.forward(input);
            L2.forward(L1.output());
            L3.forward(L2.output());
            my_nn::float_t errorp = L3.squared_error(soll);

            L3.set_weight(o*L3.in_dim() + i, weight-gc_epsilon);
            L1.forward(input);
            L2.forward(L1.output());
            L3.forward(L2.output());
            my_nn::float_t errorm = L3.squared_error(soll);


            L3.set_weight(o*L3.in_dim() + i, weight);
            L1.forward(input);
            L2.forward(L1.output());
            L3.forward(L2.output());

            L3.backward(L2,soll);
            L2.backward(L1.output(), L3);
            L1.backward(input, L2);
            my_nn::float_t gradient = L3.gc_gradient_weights(o*L3.in_dim()+i);

            //std::cout<<"Error:"<<errorp<<"  "<<errorm<<"\n";
            std::cout<<"FD - dE/d["<<o*L3.in_dim()+i<<"]="
                <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
            assert(true || fabs((errorp-errorm)/(4.0*gc_epsilon) - gradient) < 0.0001);
        }
    }

    std::cout<<"---\n";

    for (int o=0; o<L2.out_dim(); o++) {
        for (int i=0; i<L2.in_dim(); i++) {

            my_nn::float_t weight = L2.get_weight(o*L2.in_dim() + i);

            L2.set_weight(o*L2.in_dim() + i, weight+gc_epsilon);
            L1.forward(input);
            L2.forward(L1.output());
            L3.forward(L2.output());
            my_nn::float_t errorp = L3.squared_error(soll);

            L2.set_weight(o*L2.in_dim() + i, weight-gc_epsilon);
            L1.forward(input);
            L2.forward(L1.output());
            L3.forward(L2.output());
            my_nn::float_t errorm = L3.squared_error(soll);

            L2.set_weight(o*L2.in_dim() + i, weight);
            L1.forward(input);
            L2.forward(L1.output());
            L3.forward(L2.output());

            L3.backward(L2,soll);
            L2.backward(L1.output(), L3);
            L1.backward(input, L2);
            my_nn::float_t gradient = L2.gc_gradient_weights(o*L2.in_dim()+i);

            //std::cout<<"Error:"<<errorp<<"  "<<errorm<<"\n";
    
            std::cout<<"FD - dE/d["<<o*L2.in_dim()+i<<"]="
                <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
            assert(true || fabs((errorp-errorm)/(4.0*gc_epsilon) - gradient) < 0.0001);
        }
    }
}
#endif

void
fullyconnected_test2()
{
    fullyconnected_layer<sigmoid> L1(4,7);
    fullyconnected_layer<sigmoid> L2(7,5);
    fullyconnected_layer<sigmoid> L3(5,4);
    fullyconnected_layer<sigmoid> L4(4,4);

    vec_t input(4);
    vec_t soll(4);

    for(int s=0; s<1000000; s++) {

        randomize(input.begin(), input.end(), 0, 1);
        for (auto& v : input) {
            v = v<0.5 ? 0 : 1;
        }
        soll = input;
        std::reverse(soll.begin(), soll.end());

        L1.forward(input);
        L2.forward(L1.output());
        L3.forward(L2.output());

        float error = 0.0;
        for (int i=0; i<soll.size(); i++) {
            error += (soll[i]-L3.output()[i])*(soll[i]-L3.output()[i]);
        }
        
        if(s>130000) {
            print_vec("INPUT: ", input);
            print_vec("SOLL:  ", soll);
            print_vec("IST:   ", L3.output());
        }
        std::cout<< s << "   error: "<<error<<"\n";

        L3.backward(L2,soll);
        L2.backward(L1.output(), L3);
        L1.backward(input, L2);

        if(error<0.01) exit(0);
    }
}

void
cnn_training_test2()
{
    const int steps = 100000;
    mnist_reader mnist;
    mnist.read( "data/mnist/train-images-idx3-ubyte",
                "data/mnist/train-labels-idx1-ubyte", std::min(steps, 60000));

    neural_network nn;
    convolutional_layer<tan_h>  C1(28 /* in_width*/, 24 /* out_width*/, 1 /*in_fm*/,   6 /*out_fm*/);
    subsampling_layer<tan_h>    S2(24 /* in_width*/, 12 /* out_width*/, 6 /*fm*/,      2 /*block_size*/);
    convolutional_layer<tan_h>  C3(12 /* in_width*/, 10 /* out_width*/, 6 /*in_fm*/,  16 /*out_fm*/);
    subsampling_layer<tan_h>    S4(10 /* in_width*/,  5 /* out_width*/, 16 /*fm*/,     2 /*block_size*/);
    convolutional_layer<tan_h>  C5( 5 /* in_width*/,  1 /* out_width*/, 16 /*in_fm*/, 64 /*out_fm*/);
    fullyconnected_layer<tan_h> O6(64 /* in_width*/, 32 /* out_width*/);
    fullyconnected_layer<tan_h> O7(32 /* in_width*/, 10 /* out_width*/);
    nn.add_layer(&C1);
    nn.add_layer(&S2);
    nn.add_layer(&C3);
    nn.add_layer(&S4);
    nn.add_layer(&C5);
    nn.add_layer(&O6);
    nn.add_layer(&O7);
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
    C3.set_connection(connection);
    
    vec_t soll {-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8};
    int last_label = 0;
    for(int s=0; s<steps; s++) {
        
        if (s!=0 && s%1000==0) nn.set_learningrate(1.0/(s*s));
        
        int num_example = s % mnist.num_examples();

        nn.forward(mnist.image(num_example).data());
        
        soll[last_label] = -1.0;
        last_label = mnist.label(num_example);
        soll[last_label] = 1.0;

        my_nn::float_t error = O7.squared_error(soll);
        std::cout<<"Step: "<<s<<"\n";
        for (int o=0; o<soll.size(); o++)
            std::cout<<"    ["<<o<<"]: "<<soll[o]<<" vs "<<O7.output()[o]<<"\n";
        std::cout<<"    error "<<error<<"\n";

        O7.backward(O6, soll);
        O6.backward(C5.output(), O7);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);

        if ( s%10000 < 25 ) {
            std::cout<<"\n";

            Image<my_nn::float_t> img_in(28, 1*28, std::begin(mnist.image(num_example).data()), std::end(mnist.image(num_example).data()));
            Image<my_nn::float_t> img_c1(24, 6*24, std::begin(C1.output()), std::end(C1.output()));
            Image<my_nn::float_t> img_s2(12, 6*12, std::begin(S2.output()), std::end(S2.output()));
            Image<my_nn::float_t> img_c3(10,16*10, std::begin(C3.output()), std::end(C3.output()));
            Image<my_nn::float_t> img_s4( 5, 16*5, std::begin(S4.output()), std::end(S4.output()));
            Image<my_nn::float_t> img_c5( 1,   64, std::begin(C5.output()), std::end(C5.output()));
            Image<my_nn::float_t> img_o6( 1,   32, std::begin(O6.output()), std::end(O6.output()));
            Image<my_nn::float_t> img_o7( 1,   10, std::begin(O7.output()), std::end(O7.output()));

            cv::Mat img(cv::Size(400,200), CV_8UC1, 100);
            img_in.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(  0,0,img_in.width(), img_in.height())));
            img_c1.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect( 50,0,img_c1.width(), img_c1.height())));
            img_s2.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(100,0,img_s2.width(), img_s2.height())));
            img_c3.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(150,0,img_c3.width(), img_c3.height())));
            img_s4.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(200,0,img_s4.width(), img_s4.height())));
            img_c5.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(250,0,img_c5.width(), img_c5.height())));
            img_o6.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(300,0,img_o6.width(), img_o6.height())));
            img_o7.toIntensity(-1,1).exportMat().copyTo(img(cv::Rect(350,0,img_o7.width(), img_o7.height())));

            cv::imshow("cnn", img);
            while(cv::waitKey(0)!=27);
        }
    }
}

# if GRADIENT_CHECK
void
gc_cnn_training()
{
    mnist_reader mnist;
    mnist.read("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 60000);

    const my_nn::float_t gc_epsilon = 0.00001;
    std::cout.precision(16);
    std::cout.setf(std::ios::fixed, std::ios::floatfield);

    neural_network nn;
    convolutional_layer<tan_h>  C1(28 /* in_width*/, 24 /* out_width*/,  1 /*in_fm*/,  6 /*out_fm*/);
    subsampling_layer<tan_h>    S2(24 /* in_width*/, 12 /* out_width*/,  6 /*fm*/,     2 /*block_size*/);
    convolutional_layer<tan_h>  C3(12 /* in_width*/, 10 /* out_width*/,  6 /*in_fm*/, 16 /*out_fm*/);
    subsampling_layer<tan_h>    S4(10 /* in_width*/,  5 /* out_width*/, 16 /*fm*/,     2 /*block_size*/);
    convolutional_layer<tan_h>  C5( 5 /* in_width*/,  1 /* out_width*/, 16 /*in_fm*/, 12 /*out_fm*/);
    fullyconnected_layer<tan_h> O6(12 /* in_width*/, 10 /* out_width*/);
    nn.add_layer(&C1);
    nn.add_layer(&S2);
    nn.add_layer(&C3);
    nn.add_layer(&S4);
    nn.add_layer(&C5);
    nn.add_layer(&O6);

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
    
    C3.set_connection(connection);
    //Image<my_nn::float_t> img_in(,32,std::begin(),std::end(test));

    vec_t soll {-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8};
    int last_label = 0;

    int num_example = 9 % mnist.num_examples();
    nn.forward(mnist.image(num_example).data());

    soll[last_label] = -0.8;
    last_label = mnist.label(num_example);
    soll[last_label] = 0.8;

    my_nn::float_t error = O6.squared_error(soll);
    for (int o=0; o<soll.size(); o++)
        std::cout<<"    "<<soll[o]<<" vs "<<O6.output()[o]<<"\n";
    std::cout<<"    error "<<error<<"\n";

// test O6 {{{
#if 1
    std::cout<<"- O6 ---------------------------\n";
    for (int w=0; w<O6.weights().size(); w++) {

            my_nn::float_t weight = O6.get_weight(w);

            O6.set_weight(w, weight+gc_epsilon);
            nn.forward(mnist.image(num_example).data());
            my_nn::float_t errorp = O6.squared_error(soll);

            O6.set_weight(w, weight-gc_epsilon);
            nn.forward(mnist.image(num_example).data());
            my_nn::float_t errorm = O6.squared_error(soll);

            O6.set_weight(w, weight);
            nn.forward(mnist.image(num_example).data());
            O6.backward(C5, soll);
            C5.backward(S4.output(), O6);
            S4.backward(C3.output(), C5);
            C3.backward(S2.output(), S4);
            S2.backward(C1.output(), C3);
            C1.backward(mnist.image(num_example).data(), S2);
            my_nn::float_t gradient = O6.gc_gradient_weights(w);

            std::cout <<"FD - dE/d["<<w<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
                <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
    std::cout<<"- O6 bias ----------------------\n";
    for (int b=0; b<O6.bias().size(); b++) {

        my_nn::float_t bias = O6.get_bias(b);

        O6.set_bias(b, bias+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        O6.set_bias(b, bias-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        O6.set_bias(b, bias);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = O6.gc_gradient_bias(b);

        std::cout <<"FD - dE/d["<<b<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
#endif
//  }}}
// test C5 {{{
#if 1
    std::cout<<"- C5 ---------------------------\n";
    for (int w=0; w<C5.weights().size(); w++) {

        my_nn::float_t weight = C5.get_weight(w);

        C5.set_weight(w, weight+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        C5.set_weight(w, weight-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        C5.set_weight(w, weight);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = C5.gc_gradient_weights(w);

        std::cout <<"FD - dE/d["<<w<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
#endif
#if 1
    std::cout<<"- C5 bias ----------------------\n";
    for (int b=0; b<C5.bias().size(); b++) {

        my_nn::float_t bias = C5.get_bias(b);

        C5.set_bias(b, bias+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        C5.set_bias(b, bias-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        C5.set_bias(b, bias);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = C5.gc_gradient_bias(b);

        std::cout <<"FD - dE/d["<<b<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
#endif
//  }}}
// test S4 {{{
#if 1
    std::cout<<"- S4 ---------------------------\n";
    for (int w=0; w<S4.weights().size(); w++) {

        my_nn::float_t weight = S4.get_weight(w);

        S4.set_weight(w, weight+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        S4.set_weight(w, weight-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        S4.set_weight(w, weight);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = S4.gc_gradient_weights(w);

        std::cout <<"FD - dE/d["<<w<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
    std::cout<<"- S4 bias ----------------------\n";
    for (int b=0; b<S4.bias().size(); b++) {

        my_nn::float_t bias = S4.get_bias(b);

        S4.set_bias(b, bias+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        S4.set_bias(b, bias-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        S4.set_bias(b, bias);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = S4.gc_gradient_bias(b);

        std::cout <<"FD - dE/d["<<b<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
# endif
//  }}}
// test C3 {{{
#if 1
    std::cout<<"- C3 ---------------------------\n";
    for (int w=0; w<C3.weights().size(); w++) {

        my_nn::float_t weight = C3.get_weight(w);

        C3.set_weight(w, weight+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        C3.set_weight(w, weight-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        C3.set_weight(w, weight);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = C3.gc_gradient_weights(w);

        std::cout <<"FD - dE/d["<<w<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
    std::cout<<"- C3 bias ----------------------\n";
    for (int b=0; b<C3.bias().size(); b++) {

        my_nn::float_t bias = C3.get_bias(b);

        C3.set_bias(b, bias+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        C3.set_bias(b, bias-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        C3.set_bias(b, bias);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = C3.gc_gradient_bias(b);

        std::cout <<"FD - dE/d["<<b<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
# endif
//  }}}
// test S2 {{{
#if 1
    std::cout<<"- S2 ---------------------------\n";
    for (int w=0; w<S2.weights().size(); w++) {

        my_nn::float_t weight = S2.get_weight(w);

        S2.set_weight(w, weight+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        S2.set_weight(w, weight-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        S2.set_weight(w, weight);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = S2.gc_gradient_weights(w);

        std::cout <<"FD - dE/d["<<w<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
    std::cout<<"- S2 bias ----------------------\n";
    for (int b=0; b<S2.bias().size(); b++) {

        my_nn::float_t bias = S2.get_bias(b);

        S2.set_bias(b, bias+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        S2.set_bias(b, bias-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        S2.set_bias(b, bias);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = S2.gc_gradient_bias(b);

        std::cout <<"FD - dE/d["<<b<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
# endif
//  }}}
// test C1 {{{
#if 1
    std::cout<<"- C1 ---------------------------\n";
    for (int w=0; w<C1.weights().size(); w++) {

        my_nn::float_t weight = C1.get_weight(w);

        C1.set_weight(w, weight+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        C1.set_weight(w, weight-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        C1.set_weight(w, weight);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = C1.gc_gradient_weights(w);

        std::cout <<"FD - dE/d["<<w<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
    std::cout<<"- C1 bias ----------------------\n";
    for (int b=0; b<C1.bias().size(); b++) {

        my_nn::float_t bias = C1.get_bias(b);

        C1.set_bias(b, bias+gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorp = O6.squared_error(soll);

        C1.set_bias(b, bias-gc_epsilon);
        nn.forward(mnist.image(num_example).data());
        my_nn::float_t errorm = O6.squared_error(soll);

        C1.set_bias(b, bias);
        nn.forward(mnist.image(num_example).data());
        O6.backward(C5, soll);
        C5.backward(S4.output(), O6);
        S4.backward(C3.output(), C5);
        C3.backward(S2.output(), S4);
        S2.backward(C1.output(), C3);
        C1.backward(mnist.image(num_example).data(), S2);
        my_nn::float_t gradient = C1.gc_gradient_bias(b);

        std::cout <<"FD - dE/d["<<b<<"]="<<(errorp-errorm)/(4.0*gc_epsilon)<<" - "<<gradient<<" = "
            <<(errorp-errorm)/(4.0*gc_epsilon) - gradient<<"\n";
    }
# endif
//  }}}
}
#endif

void
mnist_reader_test()
{
    mnist_reader mnist;
    mnist.read("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 15);

    for (int i=0; i<mnist.num_examples(); i++) {
        cv::imshow("images_["+std::to_string(i)+"]: "+std::to_string(mnist.label(i)), mnist.image(i).exportMat());
    }
    while(cv::waitKey(0)!=27);
}

int
main(int argc, const char *argv[])
{

#if 0 && GRADIENT_CHECK
    gc_fullyconnected();
#endif

#if 0 && GRADIENT_CHECK
    gc_cnn_training();
#endif

#if 1
    cnn_training_test2();
#endif

#if 0
    fullyconnected_test();
#endif

#if 0
    fullyconnected_test2();
#endif
 
#if 0
    fullyconnected_test3();
#endif

#if 0
    mnist_reader_test();
#endif
    
    return 0;
}

