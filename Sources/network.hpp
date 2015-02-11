#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "utils.hpp"
#include "activation.hpp"

namespace my_nn {

typedef std::vector<float_t> vec_t;
typedef double float_t;

template <typename A = sigmoid>
class layer {
    
public:

    int in_dim() const { return in_dim_; }
    int out_dim() const { return out_dim_; }

    const vec_t& output() const { return output_; };
    const vec_t& weights() const { return weights_; };
    const vec_t& delta() const { return delta_; };

    typedef A ActivationFunction;
    
    layer(int in_dim, int out_dim) : in_dim_(in_dim), out_dim_(out_dim) {
        weights_.resize(in_dim*out_dim);
        output_.resize(out_dim);
        delta_.resize(out_dim);
        bias_.resize(out_dim);
        resetWeights();
    }

    void resetWeights() {
        uniform_rand(std::begin(weights_), std::end(weights_), -1.0, 1.0);
        uniform_rand(std::begin(bias_), std::end(bias_), -1.0, 1.0);
    }

    const vec_t& forward_propagate(const vec_t& in) {
        
        assert(in.size()==in_dim_);

        for (int o=0; o<output_.size(); o++) {
            float_t sum = 0.0;
            for (int i=0; i<in.size(); i++) {
                sum += in[i] * weights_[o*in_dim_ + i];
            }
            output_[o] = A_.f(sum + bias_[o]);
        }
        return output_;
    }  
    
    /* only to be used for the last layer */
    void back_propagate_last(const vec_t& in, const vec_t& soll) {
        
        assert(soll.size()==out_dim_);
        assert(in.size()==in_dim_);

        for (int o=0; o<output_.size(); o++) {
            for (int i=0; i<in.size(); i++) {
                delta_[o] = (soll[o] - output_[o]) * A_.df(output_[o]);
                weights_[o*in_dim_ + i] += learning_rate * delta_[o] * in[i];
            }
            bias_[o] += learning_rate * delta_[o] * 1.0;
        }
    }  
    
    /* to be used with every layer except the last */
    void back_propagate(const vec_t& in, layer<sigmoid>& next) {
        
        assert(in.size()==in_dim_);
        assert(next.in_dim()==out_dim_);

        for (int o=0; o<output_.size(); o++) {

            float_t sum = 0;
            for (int k=0; k<next.delta().size(); k++)
                sum += next.delta()[k] * next.weights()[o*out_dim_ + k];
            
            for (int i=0; i<in.size(); i++) {
                delta_[o] = output_[o] * A_.df(output_[o]) * sum;
                weights_[o*in_dim_ + i] += learning_rate * delta_[o] * in[i];
            }
            bias_[o] += learning_rate * delta_[o] * 1.0;
        }
    }  

private:

    ActivationFunction A_;

    int in_dim_;
    int out_dim_;
    vec_t weights_;
    vec_t bias_;

    vec_t output_;
    vec_t delta_;

    float_t learning_rate = 0.01;
};


class network {

    public:

        void add_layer(layer<sigmoid> l) {
            if (!layers.empty())
                assert(l.in_dim() == layers.back().out_dim());

            layers.push_back(l);
        }

        const std::vector< layer<sigmoid> >& get_layers() const {
            return layers;
        }

        const vec_t forward_propagate(const vec_t& in) {

            auto iter = std::begin(layers);

            vec_t output = iter->forward_propagate(in);

            while (++iter != std::end(layers)) {
                output = iter->forward_propagate(output);
            }
            return output;
        }

        void back_propagate(const vec_t in, const vec_t soll) {
            layers[2].back_propagate_last(layers[1].output(), soll);
            layers[1].back_propagate(layers[0].output(), layers[2]);
            layers[0].back_propagate(in, layers[1]);
        }

    private:

        std::vector< layer<sigmoid> > layers;

};

} /* namespace my_nn */

#endif /* NETWORK_HPP */
