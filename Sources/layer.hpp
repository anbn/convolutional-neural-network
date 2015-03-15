#ifndef LAYER_HPP
#define LAYER_HPP

#include "utils.hpp"

namespace nn {
    
class layer {
    
public:

    uint_t in_dim() const { return in_dim_; }
    uint_t out_dim() const { return out_dim_; }

    const vec_t& output() const { return output_; }
    const vec_t& delta() const { return delta_; }
    const vec_t& weights() const { return weights_; }
    const vec_t& bias() const { return bias_; }

    layer* next_layer() const { return next_layer_; }
    layer* prev_layer() const { return prev_layer_; }

    void resetWeights() {
        randomize(std::begin(weights_), std::end(weights_), -.5, .5);
        randomize(std::begin(bias_), std::end(bias_), -0.1, 0.1);
    }

    void set_next_layer(layer* next_layer) { 
        assert(next_layer!=nullptr);
        assert(out_dim_ == next_layer->in_dim());
        next_layer->set_prev_layer( this );
        next_layer_ = next_layer;
    }
    void set_prev_layer(layer* prev_layer){
        assert(prev_layer!=nullptr);
        
        prev_layer_ = prev_layer;
    }
    
    void set_learningrate(float_t l) {
        learning_rate = l;
    }

    virtual float_t in_delta_sum(uint_t fm, uint_t ix, uint_t iy) const = 0;
    virtual void forward(const vec_t& in) = 0;
    virtual void backward(const vec_t& in) = 0;

#if GRADIENT_CHECK
    vec_t gc_gradient_weights_,
          gc_gradient_bias_;

    float_t gc_gradient_weights(uint_t i) const { return gc_gradient_weights_[i]; }
    float_t gc_gradient_bias(uint_t i) const { return gc_gradient_bias_[i]; }
    float_t get_weight(uint_t i) const { return weights_[i]; }
    void    set_weight(uint_t i, float_t v) { weights_[i] = v; }
    float_t get_bias(uint_t i) const { return bias_[i]; }
    void    set_bias(uint_t i, float_t v) { bias_[i] = v; }
#endif

protected:
    
    layer(uint_t in_dim, uint_t out_dim, uint_t bias_dim, uint_t weights_dim) : in_dim_(in_dim), out_dim_(out_dim) {
        weights_.resize(weights_dim);
        output_.resize(out_dim_);
        bias_.resize(bias_dim);
        delta_.resize(out_dim_);
        resetWeights();
#if GRADIENT_CHECK
        gc_gradient_weights_.resize(weights_dim);
        gc_gradient_bias_.resize(bias_dim);
#endif
    }

    
    uint_t in_dim_;
    uint_t out_dim_;

    vec_t weights_;     /* variable size, depending on layer type */
    vec_t delta_;       /* [out_dim_] */
    vec_t bias_;        /* [feature_map] */

    vec_t output_;      /* [feature_maps * out_dim_ * out_dim_] */

    layer* next_layer_ = nullptr;
    layer* prev_layer_ = nullptr;

    float_t learning_rate = 0.0001;
};


} /* namespace nn */

#endif
