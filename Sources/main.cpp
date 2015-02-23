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


void printVec_t(std::string s, vec_t v) {
    std::cout<<s;
    for (auto e : v) {
        std::cout<<" "<<e;
    }
    std::cout<<"\n";
}


int
main(int argc, const char *argv[])
{
    vec_t input {0.0, 1.0, 1.0, 0.0};
    vec_t soll {1,0,1,0};
    vec_t output;

    network mlp;
    mlp.add_layer(layer<sigmoid>(4,8));
    mlp.add_layer(layer<sigmoid>(8,6));
    mlp.add_layer(layer<sigmoid>(6,4));

    for(int s=0; s<1000000; s++) {

        uniform_rand(input.begin(), input.end(), 0, 1);
        for (auto& v : input) {
            v = v<0.5 ? 0 : 1;
        }
        soll = input;
        std::reverse(soll.begin(), soll.end());

        output = mlp.forward_propagate(input);
        //printVec_t("INPUT: ", input);
        //printVec_t("OUTPUT:", soll);
        //printVec_t("       ", output);

        float error = 0.0;
        for (int i=0; i<soll.size(); i++)
            error += (soll[i]-output[i])*(soll[i]-output[i]);

        mlp.back_propagate(input, soll);
        
        std::cout<< s << "   error: "<<error<<"\n";

        //for (int j=0; j<mlp.get_layers()[0].weights().size(); j++)
        //    std::cout<<mlp.get_layers()[0].weights()[j]<<" ";
    }
}

