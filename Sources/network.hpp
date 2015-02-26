#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "utils.hpp"
#include "activation.hpp"
#include "image.hpp"

namespace my_nn {

typedef double float_t;
typedef std::vector<float_t> vec_t;

class layer {
    
public:

    int in_dim() const { return in_dim_; }
    int out_dim() const { return out_dim_; }

    const vec_t& output() const { return output_; };
    const vec_t& delta() const { return delta_; };
    const vec_t& weights() const { return weights_; };

    void resetWeights() {
        uniform_rand(std::begin(weights_), std::end(weights_), -1.0, 1.0);
        uniform_rand(std::begin(bias_), std::end(bias_), -1.0, 1.0);
    }

    layer(int in_dim, int out_dim) : in_dim_(in_dim), out_dim_(out_dim) {}

protected:

    size_t in_dim_;
    size_t out_dim_;

    vec_t weights_;     /* variable size, depending on layer type */
    vec_t delta_;       /* [out_dim_] */
    vec_t bias_;        /* [feature_map] */

    vec_t output_;      /* [feature_map * out_dim_ * out_dim_] */

    float_t learning_rate = 0.01;
};


//------------------------------------------------------------------------------


/* A: ActivationFunction */
template <typename ActivationFunction = sigmoid>
class convolutional_layer : public layer {

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

    
public:

    convolutional_layer(size_t in_width, size_t out_width, size_t in_feature_maps, size_t out_feature_maps)
            : layer(in_width*in_width, out_width*out_width),
            in_feature_maps_(in_feature_maps),out_feature_maps_(out_feature_maps),
            in_width_(in_width), out_width_(out_width) {
        std::cout<<"DEBUG: convolutional_layer(" <<in_width<<","<<out_width<<","<<in_feature_maps<<","<<out_feature_maps_<<")\n";
        filter_width_ = in_width-out_width+1;

        weights_.resize(in_feature_maps * out_feature_maps_ * filter_width_ * filter_width_);
        delta_.resize(in_feature_maps * out_feature_maps_ * filter_width_ * filter_width_);
        output_.resize(out_feature_maps_ * out_width_ * out_width_);
        bias_.resize(out_width_*out_width_);
        resetWeights();
    }

    void forward(size_t in_feature_maps, size_t in_width, const vec_t& in /*[in_feature_map * in_width * in_width]*/) {
        
        assert(in_width == in_width_);
        assert(in_feature_maps == in_feature_maps_);
        assert(in.size() == in_feature_maps * in_width * in_width);

        for (int fm=0; fm<out_feature_maps_; fm++) {

            for (int ox=0; ox<out_width_; ox++) {
                for (int oy=0; oy<out_width_; oy++) {

                    float_t sum = 0.0;
                    for (int in_fm=0; in_fm<in_feature_maps; in_fm++) {

                        //if (!connection_.is_connected(in_fm, fm))
                        //    continue;
                       
                        for (int fx=0; fx<filter_width_; fx++) {
                            for (int fy=0; fy<filter_width_; fy++) {
                                sum += weights_[((in_fm*out_feature_maps_ + fm)*filter_width_ + fx)*filter_width_ + fy] * 
                                          in[(in_fm*in_width_ + (ox+fx))*in_width_ + (oy+fy)];
                            }
                        }
                    }
                    output_[(fm*out_width_+ ox)*out_width_ + oy] = A_.f(sum + bias_[fm]);
                }
            }
        }
    }

private:
    
    ActivationFunction A_;

    connection_table connection_;
    size_t in_feature_maps_,out_feature_maps_;
    size_t in_width_, out_width_;
    size_t filter_width_;
};

template <typename ActivationFunction = sigmoid>
class subsampling_layer : public layer {

public:

    subsampling_layer(int in_width, int out_width, size_t out_feature_maps)
            : layer(in_width*in_width, out_width*out_width),
            out_feature_maps_(out_feature_maps),
            in_width_(in_width), out_width_(out_width) {

        assert(in_width == out_width_*2);
        
        std::cout<<"DEBUG: subsampling_layer(" <<in_width_<<","<<out_width_<<","<<out_feature_maps_<<")\n";
        weights_.resize(out_feature_maps_ * out_width_ * out_width_);
        delta_.resize(out_feature_maps_ * out_width_ * out_width_);
        output_.resize(out_feature_maps_ * out_width_ * out_width_);
        bias_.resize(out_width_ * out_width_);
        resetWeights();
    }

    void forward(size_t in_feature_maps, size_t in_width, const vec_t& in /*[in_feature_map * in_width * in_width]*/) {
        
        assert(in_feature_maps == out_feature_maps_);
        assert(in_width == out_width_*2);
        assert(in.size() == in_feature_maps * in_width_ * in_width_);

        for (int fm=0; fm<out_feature_maps_; fm++) {
            
            for (int ox=0; ox<out_width_; ox++) {
                for (int oy=0; oy<out_width_; oy++) {
                    
                    /* Average Pooling */
                       float_t sum =
                        (in[(in_width*in_width)*fm + (2*in_width*ox  ) + (2*oy  )] +
                         in[(in_width*in_width)*fm + (2*in_width*ox+1) + (2*oy  )] + 
                         in[(in_width*in_width)*fm + (2*in_width*ox  ) + (2*oy+1)] + 
                         in[(in_width*in_width)*fm + (2*in_width*ox+1) + (2*oy+1)])*0.25; /* 
                        weights_[(out_width_*out_width_)*fm + (out_width_*ox) + oy] + bias_[fm] */
                   
                    /* Max Pooling 
                       float_t sum = std::max(
                        std::max(in[(in_width*in_width)*fm + (2*in_width*ox  ) + (2*oy  )], in[(in_width*in_width)*fm + (2*in_width*ox+1) + (2*oy  )]), 
                        std::max(in[(in_width*in_width)*fm + (2*in_width*ox  ) + (2*oy+1)], in[(in_width*in_width)*fm + (2*in_width*ox+1) + (2*oy+1)]));
                    */
                    output_[(fm*out_width_ + ox)*out_width_ + oy] = A_.f( sum );
                }
            }
        }
    }
    
private:
    
    ActivationFunction A_;

    size_t out_feature_maps_;
    size_t in_width_, out_width_;
};


template <typename ActivationFunction = sigmoid>
class fullyconnected_layer : public layer {

public:

    fullyconnected_layer(size_t in_dim, size_t out_dim) : layer(in_dim, out_dim) {
        std::cout<<"DEBUG: fullyconnected_layer(" <<in_dim<<","<<out_dim<<")\n";
        weights_.resize(in_dim_ * out_dim_);
        delta_.resize(in_dim_ * out_dim_);
        output_.resize(out_dim);
        bias_.resize(out_dim);
        resetWeights();
    };


    void forward(size_t in_dim, const vec_t& in /*[in_dim]*/) {
        
        assert(in_dim == in_dim_);
        assert(in.size() == in_dim);

        for (int o=0; o<out_dim_; o++) {
            
            float_t sum=0.0;
            for (int i=0; i<in_dim_; i++) {
                sum += in[i] * weights_[ o*in_dim_ + i];
            }
            output_[o] = A_.f(sum + bias_[o]);
        }
    }

    /* backpropagation for any layer except the last one */
    void backward(const vec_t& in, const layer& next_layer) {

        assert(in.size()==in_dim_);
        assert(next_layer.in_dim()==out_dim_);

        for (int o=0; o<out_dim_; o++) {

            float_t sum = 0;
            for (int k=0; k<next_layer.out_dim(); k++)
                sum += next_layer.delta()[k] * next_layer.weights()[ o*next_layer.out_dim()+ k];

            for (int i=0; i<in_dim_; i++) {
                //delta_[o] = output_[o] * A_.df(output_[o]) * sum; // check at home
                delta_[o] = A_.df(output_[o]) * sum;
                weights_[o*in_dim_ + i] += learning_rate * delta_[o] * in[i];
            }
            bias_[o] += learning_rate * delta_[o] * 1.0;
        }
    }  

    
    /* backpropagation for the last layer */
    void backward(const layer& previous_layer, const vec_t& soll) {
        
        assert(previous_layer.out_dim()==in_dim_);
        assert(soll.size()==out_dim_);

        for (int o=0; o<out_dim_; o++) {
            for (int i=0; i<in_dim_; i++) {
                delta_[o] = (soll[o] - output_[o]) * A_.df(output_[o]);
                weights_[o*in_dim_ + i] += learning_rate * delta_[o] * previous_layer.output()[i];
            }
            bias_[o] += learning_rate * delta_[o] * 1.0;
        } 
    }

    ActivationFunction A_;
};

} /* namespace my_nn */

#endif /* NETWORK_HPP */
