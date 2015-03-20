#ifndef TEST_HPP
#define TEST_HPP

namespace nn {

#if GRADIENT_CHECK
void
gc_fullyconnected()
{
    const nn::float_t gc_epsilon = 0.00001;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(16);

    neural_network nn;
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
    mnist.read("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 10);

    const nn::float_t gc_epsilon = 0.00001;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(16);

    neural_network nn;
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
void
gc_cnn_training_fc2d()
{
    mnist_reader mnist;
    mnist.read("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 10);

    const nn::float_t gc_epsilon = 0.00001;
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(16);

    neural_network nn;
    convolutional_layer<tan_h>  C1(28 /* in_width*/, 24 /* out_width*/, 1 /*in_fm*/,   8 /*out_fm*/);
    subsampling_layer<tan_h>    S2(24 /* in_width*/, 12 /* out_width*/, 8 /*fm*/,      2 /*block_size*/);
    convolutional_layer<tan_h>  C3(12 /* in_width*/, 10 /* out_width*/, 8 /*in_fm*/,  16 /*out_fm*/);
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
    subsampling_layer<tan_h>    S4(10 /* in_width*/,  5 /* out_width*/, 16 /*fm*/,      2 /*block_size*/);
    fullyconnected_layer<tan_h> O5( 5 /* in_width*/, 10 /* out_width*/, 16 /*in_fm*/);
    nn.add_layer(&C1);
    nn.add_layer(&S2);
    nn.add_layer(&C3);
    nn.add_layer(&S4);
    nn.add_layer(&O5);

    vec_t soll {-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8};
    int last_label = 0;

    int num_example = 2 % mnist.num_examples();
    nn.forward(mnist.image(num_example).data());

    soll[last_label] = -0.8;
    last_label = mnist.label(num_example);
    soll[last_label] = 0.8;

    std::cout<<"- O5 ---------------------------\n";
    nn.gc_check_weights(mnist.image(num_example).data(), soll, &O5);
    std::cout<<"- O5 bias ----------------------\n";
    nn.gc_check_bias(mnist.image(num_example).data(), soll, &O5);

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
} /* namespace nn */

#endif /* TEST_HPP */
