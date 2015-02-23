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


//------------------------------------------------------------------------------

class connection_table {

public:
    connection_table() : rows_(0), cols_(0) {};
    connection_table(const bool *ar, size_t rows, size_t cols) : connected_(rows * cols), rows_(rows), cols_(cols) {
        std::copy(ar, ar + rows * cols, connected_.begin());
    };

    bool is_connected(int x, int y) {
        return is_empty() ? true : connected_[y * cols_ + x];
    }

    bool is_empty() {
        return rows_==0 && cols_==0;
    }    

private:
    std::vector<bool> connected_;
    size_t rows_;
    size_t cols_;
};

/* A: ActivationFunction, N: number od feature maps */
template <typename ActivationFunction = sigmoid, typename N = size_t>
class convolutional_layer {

public:
    convolutional_layer();

    void feed_forward(size_t in_feature_maps, size_t in_width, const vec_t& in /*[in_feature_map * in_width * in_width]*/) {
        
        assert(in_width == out_width_+filter_width_-1);
        assert(in.size() == in_feature_maps * in_width * in_width);

        for (int fm=0; fm<out_feature_maps_; fm++) {

            for (int ox=0; ox<out_width_; ox++) {
                for (int oy=0; oy<out_width_; oy++) {

                    float_t sum = 0.0;
                    for (int in_fm=0; in_fm<in_feature_maps; in_fm++) {

                        for (int wx=0; wx<filter_width_; wx++) {
                            for (int wy=0; wy<filter_width_; wy++) {

                                if (connection_.is_connected(in_fm, fm))
                                    sum += weights_[filter_width_*filter_width_*fm + filter_width_*wx + wy] * in[in_width*in_width*in_fm + in_width*ox + oy];
                            }
                        }
                        output_[(out_width_*out_width_)*fm + out_width_*ox + oy] = A_.f(sum + bias_[fm]);
                    }
                }
            }
        }
    }

private:
    
    ActivationFunction A_;

    connection_table connection_;
    size_t out_feature_maps_;
    size_t out_width_;
    size_t filter_width_;
    vec_t weights_;     /* [feature_map * filter_width * filter_width] */
    vec_t gradients_;   /* [feature_map * filter_width * filter_width]*/
    vec_t bias_;        /* [feature_map] */

    vec_t output_;      /* [feature_map * out_width * out_width] */
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
