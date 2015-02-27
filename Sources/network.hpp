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

    size_t in_dim() const { return in_dim_; }
    size_t out_dim() const { return out_dim_; }

    const vec_t& output() const { return output_; };
    const vec_t& delta() const { return delta_; };
    const vec_t& weights() const { return weights_; };

    void resetWeights() {
        randomize(std::begin(weights_), std::end(weights_), -1.0, 1.0);
        randomize(std::begin(bias_), std::end(bias_), -1.0, 1.0);
    }

    layer(size_t in_dim, size_t out_dim, size_t bias_dim, size_t weights_dim) : in_dim_(in_dim), out_dim_(out_dim) {
        //std::cout<<"DEBUG: layer("<<in_dim<<","<<out_dim<<","<<weights_dim<<")\n";
        weights_.resize(weights_dim);
        output_.resize(out_dim);
        bias_.resize(bias_dim);
        delta_.resize(out_dim_);
        resetWeights();
    }

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
            : layer(in_feature_maps*in_width*in_width, out_feature_maps*out_width*out_width, out_feature_maps, in_feature_maps*out_feature_maps*(in_width-out_width+1)*(in_width-out_width+1)),
            in_feature_maps_(in_feature_maps),out_feature_maps_(out_feature_maps),
            in_width_(in_width), out_width_(out_width),
            filter_width_(in_width-out_width+1) {
        std::cout<<"DEBUG: convolutional_layer(" <<in_width<<","<<out_width<<","<<in_feature_maps<<","<<out_feature_maps_<<")\n";
    }

    void forward(size_t in_feature_maps, size_t in_width, const vec_t& in /*[in_feature_map * in_width * in_width]*/) {
        
        assert(in_width == in_width_);
        assert(in_feature_maps == in_feature_maps_);
        assert(in.size() == in_feature_maps_ * in_width_ * in_width_);

        for (int fm=0; fm<out_feature_maps_; fm++) {

            for (int ox=0; ox<out_width_; ox++) {
                for (int oy=0; oy<out_width_; oy++) {

                    float_t sum = 0.0;
                    for (int in_fm=0; in_fm<in_feature_maps_; in_fm++) {

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

    void backward(const vec_t& in, const layer& next_layer) {

        assert(in.size()==in_feature_maps_ * in_width_ * in_width_);
        assert(next_layer.in_dim()==out_dim_);
        assert(next_layer.in_dim()==next_layer.out_dim()*4); /* true for any 2*2 subsampling_layer */

        for (int fm=0; fm<out_feature_maps_; fm++) {
            
            float_t sum = 0.0;
            for (int ox=0; ox<out_width_; ox++) {
                for (int oy=0; oy<out_width_; oy++) {
                    int oindex = (fm*out_width_+ ox)*out_width_ + oy;
                    //delta_[oindex] = A_.df(output_[oindex]) * 
                    //    next_layer.delta()[(in_width*in_width)*fm + (2*in_width*ox) + (2*oy)] * next_layer.weights()[o];
                }
            }



            for (int o=0; o<out_dim_; o++) {

                sum += delta_[o];
                
                for (int i=0; i<in_dim_; i++) {
                    //weights_[]
                }
                
            }
            //bias_[o] = learning_rate * sum;
        }

    }

private:
    
    ActivationFunction A_;

    connection_table connection_;
    size_t in_feature_maps_, out_feature_maps_;
    size_t in_width_, out_width_;
    size_t filter_width_;
};

template <typename ActivationFunction = sigmoid>
class subsampling_layer : public layer {

public:

    subsampling_layer(size_t in_width, size_t out_width, size_t feature_maps)
            : layer(feature_maps*in_width*in_width, feature_maps*out_width*out_width, feature_maps, feature_maps*out_width*out_width),
            feature_maps_(feature_maps),
            in_width_(in_width), out_width_(out_width) {

        assert(in_width == out_width*2);
        
        std::cout<<"DEBUG: subsampling_layer(" <<in_width_<<","<<out_width_<<","<<feature_maps_<<")\n";
    }

    void forward(size_t in_feature_maps, size_t in_width, const vec_t& in /*[in_feature_map * in_width * in_width]*/) {
        
        assert(in_feature_maps == feature_maps_);
        assert(in_width == out_width_*2);
        assert(in.size() == feature_maps_ * in_width_ * in_width_);

        for (int fm=0; fm<feature_maps_; fm++) {
            
            for (int ox=0; ox<out_width_; ox++) {
                for (int oy=0; oy<out_width_; oy++) {

                    float_t sum =   in[(fm*in_width + (2*ox  ))*in_width + (2*oy  )] +
                                    in[(fm*in_width + (2*ox+1))*in_width + (2*oy  )] + 
                                    in[(fm*in_width + (2*ox  ))*in_width + (2*oy+1)] + 
                                    in[(fm*in_width + (2*ox+1))*in_width + (2*oy+1)];
                   
                    /* Average Pooling, if all weights == 0.25 for 2*2 block */
                    /* Max Pooling 
                       float_t sum = std::max(
                        std::max(in[(in_width*in_width)*fm + (2*in_width*ox  ) + (2*oy  )], in[(in_width*in_width)*fm + (2*in_width*ox+1) + (2*oy  )]), 
                        std::max(in[(in_width*in_width)*fm + (2*in_width*ox  ) + (2*oy+1)], in[(in_width*in_width)*fm + (2*in_width*ox+1) + (2*oy+1)]));
                    */
                    output_[(fm*out_width_ + ox)*out_width_ + oy] = A_.f( weights_[fm]*sum + bias_[fm] );
                }
            }
        }
    }
    
private:
    
    ActivationFunction A_;

    size_t feature_maps_;   /* feature_maps_ == in_feature_maps_ == out_feature_maps_ */
    size_t in_width_, out_width_;
};


template <typename ActivationFunction = sigmoid>
class fullyconnected_layer : public layer {

public:

    fullyconnected_layer(size_t in_dim, size_t out_dim)
        : layer(in_dim, out_dim, out_dim, in_dim*out_dim) {
        std::cout<<"DEBUG: fullyconnected_layer(" <<in_dim<<","<<out_dim<<")\n";
    };


    void forward(size_t in_dim, const vec_t& in /*[in_dim]*/) {
        
        assert(in_dim == in_dim_);
        assert(in.size() == in_dim);

        for (int o=0; o<out_dim_; o++) {
            
            float_t sum=0.0;
            for (int i=0; i<in_dim_; i++) {
                sum += in[i] * weights_[o*in_dim_ + i];
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
            
            delta_[o] = A_.df(output_[o]) * sum; //vs. delta_[o] = output_[o] * A_.df(output_[o]) * sum; // check at home
           
            for (int i=0; i<in_dim_; i++) {
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

            delta_[o] = (soll[o] - output_[o]) * A_.df(output_[o]);
            
            for (int i=0; i<in_dim_; i++) {
                weights_[o*in_dim_ + i] += learning_rate * delta_[o] * previous_layer.output()[i];
            }
            bias_[o] += learning_rate * delta_[o] * 1.0;
        } 
    }

    /* TODO
    template<typename BackwardFunction>
    vec_t finite_difference_testing(int i, int o, BackwardFunction _Bf){
        vec_t gradients;

    }*/

    ActivationFunction A_;
};

} /* namespace my_nn */

#endif /* NETWORK_HPP */
