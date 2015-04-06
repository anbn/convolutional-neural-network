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

    void reset_weights(float_t sample) {
        get_random(std::begin(weights_), std::end(weights_), -1.0/sqrt(sample), 1.0/sqrt(sample));
        std::cout<<"  sample from "<<sample<<" ["<<-sqrt(1.0/sample)<<", "<<sqrt(1.0/sample)<<"]\n";
        std::fill(std::begin(bias_), std::end(bias_), 0.0);
        std::fill(std::begin(batch_gradient_weights_), std::end(batch_gradient_weights_), 0.0);
        std::fill(std::begin(batch_gradient_bias_), std::end(batch_gradient_bias_), 0.0);
    }

    void set_next_layer(layer* next_layer) { 
        assert(next_layer!=nullptr);
#if VERBOSE
        std::cout<<"set_next_layer(...): "<<out_dim_<<"=="<<next_layer->in_dim()<<"\n";
#endif
        assert(out_dim_ == next_layer->in_dim());
        next_layer->set_prev_layer( this );
        next_layer_ = next_layer;
    }
    void set_prev_layer(layer* prev_layer){
        assert(prev_layer!=nullptr);
        prev_layer_ = prev_layer;
    }
    
    virtual float_t in_delta_sum(uint_t fm, uint_t ix, uint_t iy) const = 0;
    virtual void forward(const vec_t& in) = 0;
    virtual void backward(const vec_t& in, bool is_update) = 0;

#if GRADIENT_CHECK
    float_t gc_gradient_weights(uint_t i) const { return gc_gradient_weights_[i]; }
    float_t gc_gradient_bias(uint_t i) const { return gc_gradient_bias_[i]; }
    float_t get_weight(uint_t i) const { return weights_[i]; }
    void    set_weight(uint_t i, float_t v) { weights_[i] = v; }
    float_t get_bias(uint_t i) const { return bias_[i]; }
    void    set_bias(uint_t i, float_t v) { bias_[i] = v; }
#endif

#if TRAINING_MOMENTUM
    float_t learning_rate() { return learning_rate_; }
    float_t momentum() { return momentum_; }
    float_t decay() { return decay_; }

    void set_learning_rate(float_t v) { learning_rate_ = v; }
    void set_momentum(float_t v) { momentum_ = v; }
    void set_decay(float_t v) { decay_ = v; }
#endif

protected:
    
    layer(uint_t in_dim, uint_t out_dim, uint_t bias_dim, uint_t weights_dim) : in_dim_(in_dim), out_dim_(out_dim) {
        weights_.resize(weights_dim);
        output_.resize(out_dim_);
        bias_.resize(bias_dim);
        delta_.resize(out_dim_);
        batch_gradient_weights_.resize(weights_dim);
        batch_gradient_bias_.resize(bias_dim);

        dropout_sample_.resize(in_dim);

#if GRADIENT_CHECK
        gc_gradient_weights_.resize(weights_dim);
        gc_gradient_bias_.resize(bias_dim);
#endif        

#if TRAINING_MOMENTUM
        mom_weights_.resize(weights_dim);
        mom_bias_.resize(bias_dim);
        
#elif TRAINING_ADADELTA
        ad_acc_gradient_weights_.resize(weights_dim);
        ad_acc_updates_weights_.resize(weights_dim);
        ad_acc_gradient_bias_.resize(bias_dim);
        ad_acc_updates_bias_.resize(bias_dim);
        std::fill(std::begin(ad_acc_gradient_weights_),std::end(ad_acc_gradient_weights_), 0.0);
        std::fill(std::begin(ad_acc_updates_weights_),std::end(ad_acc_updates_weights_), 0.0);
        std::fill(std::begin(ad_acc_gradient_bias_),std::end(ad_acc_gradient_bias_), 0.0);
        std::fill(std::begin(ad_acc_updates_bias_),std::end(ad_acc_updates_bias_), 0.0);
#endif
    }
    
    uint_t in_dim_;
    uint_t out_dim_;

    vec_t weights_;     /* variable size, depending on layer type */
    vec_t bias_;        /* [feature_map] */
    vec_t delta_;       /* [out_dim_] */

    vec_t output_;      /* [feature_maps * out_dim_ * out_dim_] */

    layer* next_layer_ = nullptr;
    layer* prev_layer_ = nullptr;

    vec_t batch_gradient_weights_;
    vec_t batch_gradient_bias_;

    float_t dropout_prob_;
    vec_t dropout_sample_;

    void sample_dropout() {
        get_random(std::begin(dropout_sample_), std::end(dropout_sample_), 0.0, 1.0);
    }

#if GRADIENT_CHECK
    vec_t gc_gradient_weights_,
          gc_gradient_bias_;
#endif

#if TRAINING_MOMENTUM
    /* learning parameters for SGD with momentum and decay */
    float_t learning_rate_ = 0.01;
    float_t momentum_ = 0.9;
    float_t decay_ = 0.001;

    vec_t mom_weights_;
    vec_t mom_bias_;
    
#elif TRAINING_ADADELTA
    /* learning parameters for SGD with adadelta */
    float_t ad_epsilon_ = 1e-6;
    float_t ad_ro_ = 0.95;

    vec_t ad_acc_gradient_weights_;
    vec_t ad_acc_updates_weights_;
    vec_t ad_acc_gradient_bias_;
    vec_t ad_acc_updates_bias_;
#endif
};


class output_layer : public layer {

public:
    
    output_layer(uint_t in_dim, uint_t out_dim, uint_t bias_dim, uint_t weights_dim) 
        : layer(in_dim, out_dim, bias_dim, weights_dim)
    {}

    virtual float_t error(const vec_t& soll) const = 0;

    void set_soll( const vec_t* soll ) {
        soll_ = soll;
    }
    
protected:

    const vec_t* soll_ = nullptr;
};


} /* namespace nn */

#endif /* LAYER_HPP */
