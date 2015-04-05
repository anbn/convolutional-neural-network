#include <iostream>
#include <fstream>

#include <opencv2/highgui/highgui.hpp>

#define GRADIENT_CHECK  1

#define POOLING_AVG         1
#define POOLING_MAX         0 // GRADIENT_CHECK fails!


#define TRAINING_MOMENTUM   1
#define TRAINING_ADADELTA   0

#define BATCH_SIZE          1

#define VERBOSE             0

#include "network.hpp"
#include "image.hpp"
#include "mnist_reader.hpp"
#include "orl_reader.hpp"
#include "test.hpp"
#include "gnuplot.hpp"

using namespace nn;

void print_vec(vec_t v) {
    for (auto e : v)
        std::cout<<" "<<e;
    std::cout<<"\n";
}


void
fullyconnected_test()
{
    gnuplot gp("error.txt");
    gp.init_plot("error_fct");
    
    fullyconnected_layer<tan_h> L1(4,8);
    fullyconnected_layer<tan_h> L2(8,6);
    fullyconnected_layer<tan_h> L3(6,5);
    fullyconnected_layer<tan_h> L4(5,4);

    neural_network nn;
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
            std::cout<<"INPUT "; print_vec(input);
            std::cout<<"SOLL  "; print_vec(soll);
            std::cout<<"IST   "; print_vec(nn.output());
        }
        std::cout<< s << "   error: "<<error<<" \t("<<moving_error<<")\n";
        gp.plot_point(error);

        nn.backward(input, soll);

        moving_error = 0.9*moving_error + 0.1*error;
        if(moving_error<0.01) exit(0);
    }
}


nn::float_t mnist_rate(neural_network &nn)
{
    static mnist_reader mnist_test;
    if (mnist_test.num_examples()==0)
        mnist_test.read("data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", 10000);
    
    int c=0, w=0;
    for (int n=0; n<mnist_test.num_examples(); n++) {
        nn.forward(mnist_test.image(n).data());
        int max = 0;
        for (int o=0; o<10; o++) {
            if(nn.output()[max]<nn.output()[o])
                max = o;
        }
        if(max==mnist_test.label(n)) {
            c++;
        } else {
            w++;
        }

    }
    nn::float_t rate = ((nn::float_t) c)/(c+w);
    std::cout<<"mnist_rate(...): "<<c<<" correct, "<<w<<" false ("<<rate<<")\n";
    return rate;
}

void
cnn_training_test_mnist()
{
    const int steps = 100000;
    mnist_reader mnist_train;
    mnist_train.read("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 0);
    gnuplot gp("error.txt");
    gp.init_plot("error");

    neural_network nn;
    convolutional_layer<relu>   C1(28 /* in_width*/, 24 /* out_width*/, 1 /* in_fm*/,   8 /* out_fm*/);
    subsampling_layer<identity> S2(24 /* in_width*/, 12 /* out_width*/, 8 /* fm*/,      2 /* block_size*/);
    convolutional_layer<relu>   C3(12 /* in_width*/, 10 /* out_width*/, 8 /* in_fm*/,  16 /* out_fm*/);
#define _ false
#define X true
    const bool connection[] = {
        X, _, _, _, X, X, X, _, _, X, X, X, X, _, X, X,
        X, X, _, _, _, X, X, X, _, _, X, X, X, X, _, X,
        X, X, X, _, _, _, X, X, X, _, _, X, _, X, X, X,
        _, X, X, X, _, _, X, X, X, X, _, _, X, _, X, X,
        _, _, X, X, X, _, _, X, X, X, X, _, X, X, _, X,
        _, _, _, X, X, X, _, _, X, X, X, X, _, X, X, X,
        _, _, _, _, X, X, X, _, _, X, X, X, X, _, X, X,
        _, _, _, _, _, X, X, X, _, _, X, X, X, X, _, X
    };
#undef _
#undef X
    C3.set_connection(connection, 8*16);
    subsampling_layer<identity>    S4(10 /* in_width*/,  5 /* out_width*/, 16 /* fm*/,      2 /* block_size*/);
    fullyconnected_layer<identity> O5( 5 /* in_width*/, 10 /* out_width*/, 16 /* in_fm*/);
    softmax_layer M6( 10 /* in_dim */);
    nn.add_layer(&C1);
    nn.add_layer(&S2);
    nn.add_layer(&C3);
    nn.add_layer(&S4);
    nn.add_layer(&O5);
    nn.add_layer(&M6);
    
#if TRAINING_MOMENTUM
    nn.set_learning_rate(0.00085);
#endif
    vec_t soll {0,0,0,0,0,0,0,0,0,0};
    int last_label = 0;
    for(int s=0; s<steps; s++) {
        
        
#if TRAINING_MOMENTUM
        if (s!=0 && s%1000==0)
            nn.set_learning_rate(nn.learning_rate()*0.85);
#endif
        
        int num_example = s % mnist_train.num_examples();

        int shift_x = get_random(-2.0,2.0);
        int shift_y = get_random(-2.0,2.0);
        nn.forward(mnist_train.image_shifted(num_example, shift_x, shift_y).data());
        
        soll[last_label] =  0.0;
        last_label = mnist_train.label(num_example);
        soll[last_label] =  1.0;

        nn::float_t error = nn.error(soll);

        std::cout<<"Step "<<s<<"\n";
        for (int o=0; o<soll.size(); o++)
            std::cout<<"   ["<<o<<"]: "<<soll[o]<<" vs "<<nn.output()[o]<<"\n";
        std::cout<<"   error "<<error<<"\n";

        gp.plot_point(error);
        nn.backward(mnist_train.image(num_example).data(), soll);

#if 0
        /* test on mnist test set */
        if( s!=0 && s%10000==0 ) {
            mnist_rate(nn);
        }
#else
        if ( s%5000 < 100 ) {

            cv::Mat img(cv::Size(400,200), CV_8UC1, 100);
            Image<nn::float_t> img_in(28, 1*28, std::begin(mnist_train.image(num_example).data()), std::end(mnist_train.image(num_example).data()));
            Image<nn::float_t> img_c1(24, 8*24, std::begin(C1.output()), std::end(C1.output()));
            Image<nn::float_t> img_s2(12, 8*12, std::begin(S2.output()), std::end(S2.output()));
            Image<nn::float_t> img_c3(10,16*10, std::begin(C3.output()), std::end(C3.output()));
            Image<nn::float_t> img_s4( 5, 16*5, std::begin(S4.output()), std::end(S4.output()));
            Image<nn::float_t> img_o5( 1,   10, std::begin(O5.output()), std::end(O5.output()));
            Image<nn::float_t> img_m6( 1,   10, std::begin(M6.output()), std::end(M6.output()));
            Image<nn::float_t> img_so( 1,   10, std::begin(soll), std::end(soll));

            img_in.toIntensity().exportMat().copyTo(img(cv::Rect(  0,0,img_in.width(), img_in.height())));
            img_c1.toIntensity().exportMat().copyTo(img(cv::Rect( 50,0,img_c1.width(), img_c1.height())));
            img_s2.toIntensity().exportMat().copyTo(img(cv::Rect(100,0,img_s2.width(), img_s2.height())));
            img_c3.toIntensity().exportMat().copyTo(img(cv::Rect(150,0,img_c3.width(), img_c3.height())));
            img_s4.toIntensity().exportMat().copyTo(img(cv::Rect(200,0,img_s4.width(), img_s4.height())));
            img_o5.toIntensity().exportMat().copyTo(img(cv::Rect(250,0,img_o5.width(), img_o5.height())));
            img_m6.toIntensity().exportMat().copyTo(img(cv::Rect(251,0,img_m6.width(), img_m6.height())));
            img_so.toIntensity().exportMat().copyTo(img(cv::Rect(252,0,img_so.width(), img_so.height())));
            cv::imshow("cnn", img);

#if 0
            /* show weights */
            cv::Mat img_weights(cv::Size(400,500), CV_8UC1, 100);
            Image<nn::float_t> img_weights_c1(5, 8*5, std::begin(C1.weights()), std::end(C1.weights()));
            Image<nn::float_t> img_weights_c3(3,16*3*8, std::begin(C3.weights()), std::end(C3.weights()));
            
            img_weights_c1.toIntensity(-2,2).exportMat().copyTo(img_weights(cv::Rect( 50,0,img_weights_c1.width(), img_weights_c1.height())));
            img_weights_c3.toIntensity(-2,2).exportMat().copyTo(img_weights(cv::Rect(150,0,img_weights_c3.width(), img_weights_c3.height())));
            cv::imshow("cnn_weights", img_weights);
#endif            
            while(cv::waitKey(0)!=27);
        }
#endif
    }
    gp.finish_plot();
}

int
main(int argc, const char *argv[])
{
    std::cout<<"NeuralNetwork, compiled "<<__DATE__<<" at "<<__TIME__<<"\n";
    std::cout<<"  BATCH_SIZE is "<<BATCH_SIZE<<"\n";

#if POOLING_AVG
    std::cout<<"  POOLING_AVG is enabled.\n";
#elif POOLING_MAX
    std::cout<<"  POOLING_MAX is enabled.\n";
#else
    std::cout<<"Warning: no pooling method enabled.\n";
#endif

#if TRAINING_MOMENTUM
    std::cout<<"  TRAINING_MOMENTUM is enabled.\n";
#elif TRAINING_ADADELTA
    std::cout<<"  TRAINING_ADADELTA is enabled.\n";
#else
    std::cout<<"Warning: no training method enabled.\n";
#endif
    
#if GRADIENT_CHECK
    std::cout<<"  GRADIENT_CHECK enabled.\n";
    assert(BATCH_SIZE==1);
    //gc_fullyconnected();
    //gc_cnn_training();
    gc_cnn_training_fc2d();
    return 0;
#endif


    //fullyconnected_test();
    cnn_training_test_mnist();

    //orl_reader_test();
    //mnist_reader_test();
    
    return 0;
}
