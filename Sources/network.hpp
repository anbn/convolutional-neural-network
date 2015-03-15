#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <list>

#include "utils.hpp"
#include "activation.hpp"

#include "image.hpp"
#include "mnist_reader.hpp"

#define GRADIENT_CHECK  0
#define VERBOSE         0

namespace my_nn {

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

//------------------------------------------------------------------------------

template <typename ActivationFunction = sigmoid>
class fullyconnected_layer : public layer {

public:

    fullyconnected_layer(uint_t in_dim, uint_t out_dim)
        : layer(in_dim, out_dim, out_dim, in_dim*out_dim)
    {
        std::cout<<"DEBUG: fullyconnected_layer(" <<in_dim<<","<<out_dim<<")\n";
    }


    void forward(const vec_t& in /*[in_dim]*/) {
        
        assert(in.size() == in_dim_);

        for (uint_t o=0; o<out_dim_; o++) {
            
            float_t sum=0.0;
            for (uint_t i=0; i<in_dim_; i++) {
                sum += in[i] * weights_[o*in_dim_ + i];
            }
            output_[o] = A_.f(sum + bias_[o]);
        }
    }

    float_t squared_error(const vec_t& soll) {

        assert(soll.size()==out_dim_);
        
        float_t error = 0.0;
        for(uint_t o=0; o<out_dim_; o++)
            error += (soll[o]-output_[o])*(soll[o]-output_[o]);
        return error;
    }

    float_t in_delta_sum(uint_t fm, uint_t ix, uint_t iy) const {
        
        assert(fm < in_dim_);
        assert(ix==0 && iy==0);
        
        float_t sum = 0.0;
        for(uint_t o=0; o<out_dim_; o++)
            sum += delta_[o] * weights_[o*in_dim_ + fm];
        return sum;
    }

    /* backpropagation for any layer */
    void backward(const vec_t& in) {
#if VERBOSE
        std::cout<<"(backwardfully) ";
#endif

        assert(in.size()==in_dim_);

        for (uint_t o=0; o<out_dim_; o++) {

            if (next_layer_==nullptr) { /* is the last layer */
                delta_[o] = (output_[o]-(*soll_)[o]) * A_.df(output_[o]); 
            } else {               /* not the last layer */
                delta_[o] = A_.df(output_[o]) * next_layer_->in_delta_sum(o,0,0);
            }
            for (uint_t i=0; i<in_dim_; i++) {
#if GRADIENT_CHECK
                gc_gradient_weights_[o*in_dim_ + i] = delta_[o] * in[i]; 
#else
                weights_[o*in_dim_ + i] -= learning_rate * delta_[o] * in[i];
#endif
            }
#if GRADIENT_CHECK
            gc_gradient_bias_[o] = delta_[o] * 1.0;
#else
            bias_[o] -= learning_rate * delta_[o] * 1.0;
#endif
        }
    } 

    void set_soll( const vec_t* soll ) {
        soll_ = soll;
    }

    ActivationFunction A_;

private:
    const vec_t* soll_ = nullptr;
};


/* A: ActivationFunction */
template <typename ActivationFunction = sigmoid>
class convolutional_layer : public layer {

public:

    convolutional_layer(uint_t in_width, uint_t out_width, uint_t in_feature_maps, uint_t out_feature_maps)
            : layer(in_feature_maps*in_width*in_width, out_feature_maps*out_width*out_width, out_feature_maps, in_feature_maps*out_feature_maps*(in_width-out_width+1)*(in_width-out_width+1)),
            in_feature_maps_(in_feature_maps),out_feature_maps_(out_feature_maps),
            in_width_(in_width), out_width_(out_width),
            filter_width_(in_width-out_width+1)
    {
        std::cout<<"DEBUG: convolutional_layer(" <<in_width<<","<<out_width<<","<<in_feature_maps<<","<<out_feature_maps_<<")\n";
        connected_.resize(in_feature_maps_*out_feature_maps);
        std::fill(std::begin(connected_), std::end(connected_), true);
    }

    uint_t out_feature_maps() const { return out_feature_maps_; }
    uint_t in_feature_maps() const { return in_feature_maps_; }
    uint_t filter_width() const { return filter_width_; }

    void set_connection(const bool *p, uint_t size) {
        assert(size == in_feature_maps_*out_feature_maps_);
        std::copy(p, p + in_feature_maps_*out_feature_maps_, connected_.begin());
    }
    
    inline bool is_connected(uint_t i, uint_t o) const {
        return connected_[i * out_feature_maps_ + o];
    }

    void forward(const vec_t& in /*[in_feature_map * in_width * in_width]*/) {
        
        assert(in.size() == in_feature_maps_ * in_width_ * in_width_);
        assert(weights_.size() == in_feature_maps_*out_feature_maps_*filter_width_*filter_width_);

        for (uint_t out_fm=0; out_fm<out_feature_maps_; out_fm++) {

            for (uint_t ox=0; ox<out_width_; ox++) {
                for (uint_t oy=0; oy<out_width_; oy++) {

                    float_t sum = 0.0;
                    for (uint_t in_fm=0; in_fm<in_feature_maps_; in_fm++) {

                        if (!is_connected(in_fm, out_fm))
                            continue;

                        for (uint_t fx=0; fx<filter_width_; fx++) {
                            for (uint_t fy=0; fy<filter_width_; fy++) {
                                sum += weights_[((in_fm*out_feature_maps_ + out_fm)*filter_width_ + fx)*filter_width_ + fy] *
                                    in[(in_fm*in_width_ + (ox+fx))*in_width_ + (oy+fy)];
                            }
                        }
                    }
                    output_[(out_fm*out_width_+ ox)*out_width_ + oy] = A_.f(sum + bias_[out_fm]);
                }
            }
        }
    }

    /* convolutional_layer */
    float_t in_delta_sum(uint_t in_fm, uint_t ix, uint_t iy) const {

        assert(in_fm < in_feature_maps_);
        assert(ix < in_width_);
        assert(iy < in_width_);

        float_t sum = 0.0;
        for (uint_t out_fm=0; out_fm<out_feature_maps_; out_fm++) {
            
            if(!is_connected(in_fm, out_fm))
                continue;

            for(uint_t fx = 0; fx<filter_width_; fx++ ) {
                for(uint_t fy = 0; fy<filter_width_; fy++ ) {
                    if ((ix-fx>=0) && (iy-fy>=0) && (ix-fx<out_width_) && (iy-fy<out_width_)) {
                        sum += delta_[(out_fm*out_width_ + (ix-fx))*out_width_ + (iy-fy)] * 
                                weights_[((in_fm*out_feature_maps_ + out_fm)*filter_width_ + fx)*filter_width_ + fy];
                    }
                }
            }
        }
        return sum;
    }
    
    /* convolutional_layer */
    void backward(const vec_t& in) {
#if VERBOSE
        std::cout<<"(backwardconv) ";
#endif
        assert(in.size()==in_feature_maps_ * in_width_ * in_width_);
        assert(weights_.size() == in_feature_maps_*out_feature_maps_*filter_width_*filter_width_);


        for (uint_t out_fm=0; out_fm<out_feature_maps_; out_fm++) {
            
            float_t sum_delta = 0.0;
            for (uint_t ox=0; ox<out_width_; ox++) {
                for (uint_t oy=0; oy<out_width_; oy++) {
                    uint_t out_index = (out_fm*out_width_+ ox)*out_width_ + oy;
                    delta_[out_index] = A_.df(output_[out_index]) * next_layer_->in_delta_sum(out_fm, ox, oy);
                    sum_delta += delta_[out_index];
                }
            }
#if GRADIENT_CHECK
            gc_gradient_bias_[out_fm] = sum_delta;
#else
            bias_[out_fm] -= learning_rate * sum_delta;
#endif       
            for (uint_t in_fm=0; in_fm<in_feature_maps_; in_fm++) {

                if (!is_connected(in_fm, out_fm))
                    continue;

                for (uint_t fx=0; fx<filter_width_; fx++) {
                    for (uint_t fy=0; fy<filter_width_; fy++) {

                        float_t sum = 0.0;
                        for (uint_t ox=0; ox<out_width_; ox++) {
                            for (uint_t oy=0; oy<out_width_; oy++) {
                                sum += delta_[(out_fm*out_width_+ ox)*out_width_ + oy] *
                                    in[(in_fm*in_width_ + (ox+fx))*in_width_ + (oy+fy)];
                            }
                        }
#if GRADIENT_CHECK
                        gc_gradient_weights_[((in_fm*out_feature_maps_ + out_fm)*filter_width_ + fx)*filter_width_ + fy] = sum;
#else
                        weights_[((in_fm*out_feature_maps_ + out_fm)*filter_width_ + fx)*filter_width_ + fy] -= learning_rate * sum;
#endif
                    }
                }
            }
        }
    }

private:
    
    ActivationFunction A_;

    uint_t in_feature_maps_, out_feature_maps_;
    uint_t in_width_, out_width_;
    uint_t filter_width_;
    std::vector<bool> connected_;
};


template <typename ActivationFunction = sigmoid>
class subsampling_layer : public layer {

public:

    subsampling_layer(uint_t in_width, uint_t out_width, uint_t feature_maps, uint_t block_size)
            : layer(feature_maps*in_width*in_width, feature_maps*out_width*out_width, feature_maps, feature_maps),
            feature_maps_(feature_maps),
            block_size_(block_size),
            in_width_(in_width), out_width_(out_width)
    {

        std::cout<<"DEBUG: subsampling_layer(" <<in_width_<<","<<out_width_<<","<<feature_maps_<<", "<<block_size_<<")\n";
        
        assert(in_width == out_width*block_size);
    }

    void forward(const vec_t& in /*[in_feature_map * in_width_ * in_width_]*/) {
        
        assert(in.size() == feature_maps_ * in_width_ * in_width_);

        for (uint_t fm=0; fm<feature_maps_; fm++) {
            
            for (uint_t ox=0; ox<out_width_; ox++) {
                for (uint_t oy=0; oy<out_width_; oy++) {

                    float_t sum = 0.0;
                    for (uint_t bx=0; bx<block_size_; bx++) {
                        for (uint_t by=0; by<block_size_; by++) {
                            sum += in[(fm*in_width_ + (block_size_*ox+bx))*in_width_ + (block_size_*oy+by)];
                        }
                    }

                    output_[(fm*out_width_ + ox)*out_width_ + oy] = A_.f( weights_[fm]*sum + bias_[fm] );
                }
            }
        }
    }
    
    float_t in_delta_sum(uint_t fm, uint_t ix, uint_t iy) const {

        assert(fm < feature_maps_);
        assert(ix < in_width_);
        assert(iy < in_width_);
        
        return delta_[(fm*out_width_ + ix/2)*out_width_ + iy/2] * weights_[fm];
    }
    
    void backward(const vec_t& in) {
#if VERBOSE
        std::cout<<"(backwardsub) ";
#endif
        assert(in.size()==in_dim_);

        for (uint_t fm=0; fm<feature_maps_; fm++) {

            float_t sum = 0.0;
            float_t sum_delta = 0.0;
            for (uint_t ox=0; ox<out_width_; ox++) {
                for (uint_t oy=0; oy<out_width_; oy++) {

                    const uint_t out_index = (fm*out_width_+ ox)*out_width_ + oy;
                    delta_[out_index] = A_.df(output_[out_index]) * next_layer_->in_delta_sum(fm,ox,oy);
                    sum_delta += delta_[out_index]; 
                    
                    float_t sum_block = 0.0;
                    for (uint_t bx=0; bx<block_size_; bx++) {
                        for (uint_t by=0; by<block_size_; by++) {
                            sum_block += in[(fm*in_width_ + (block_size_*ox+bx  ))*in_width_ + (block_size_*oy+by  )];
                        }
                    }
                    sum += sum_block * delta_[out_index];
                }
            }

#if GRADIENT_CHECK
            gc_gradient_bias_[fm] = sum_delta;
            gc_gradient_weights_[fm] = sum;
#else
            bias_[fm] -= learning_rate*sum_delta;
            weights_[fm] -= learning_rate*sum;
#endif
        }
    }

private:
    
    ActivationFunction A_;

    uint_t feature_maps_;   /* feature_maps_ == in_feature_maps_ == out_feature_maps_ */
    uint_t block_size_;
    uint_t in_width_, out_width_;
};


template<typename ActivationFunction>
class neural_network {

public:
    
    uint_t in_dim() const { assert(first_layer_!=nullptr); return first_layer_->in_dim(); }
    uint_t out_dim() const { assert(last_layer_!=nullptr); return last_layer_->out_dim(); }

    const vec_t& output() const {
        assert(last_layer_!=nullptr);
        return last_layer_->output();
    }

    void set_learningrate(float_t learning_rate) {
        assert(first_layer_!=nullptr);

        layer* l = first_layer_;
        do {
            l->set_learningrate(learning_rate);
            l = l->next_layer();
        } while(l!=nullptr); 
    }
    
    void add_layer(layer* l) {
        assert(l!=nullptr);

        if (last_layer_==nullptr) {
            first_layer_ = l;
        } else {
            last_layer_->set_next_layer(l);
        }
        last_layer_ = l;
    }

    float_t squared_error(const vec_t& soll) {
        fullyconnected_layer<ActivationFunction> *last = dynamic_cast<fullyconnected_layer<ActivationFunction>*>(last_layer_);
        assert(last != nullptr);
        return last->squared_error(soll);
    }

    void forward(const vec_t& in) {
        
        assert(first_layer_!=nullptr);

        layer* l = first_layer_;
        const vec_t* input = &in;

        do {
            l->forward(*input);
            input = &l->output();
            l = l->next_layer();
        } while(l!=nullptr); 
    }

    void backward(const vec_t& in, const vec_t& soll) {
       
        fullyconnected_layer<ActivationFunction> *last = dynamic_cast<fullyconnected_layer<ActivationFunction>*>(last_layer_);
        assert(last != nullptr);

        last->set_soll(&soll);
        
        layer* l = last_layer_;

        do {
            l->backward(l->prev_layer()==nullptr? in : l->prev_layer()->output());
            l = l->prev_layer();
        } while(l!=nullptr);
    }

# if GRADIENT_CHECK
    void gc_check_weights(const vec_t& in, const vec_t& soll, layer* gc_layer) {
       
        const my_nn::float_t gc_epsilon = 0.00001;
        
        for (int w=0; w<gc_layer->weights().size(); w++) {

            float_t weight = gc_layer->get_weight(w);

            gc_layer->set_weight(w, weight+gc_epsilon);
            this->forward(in);
            float_t errorp = this->squared_error(soll);

            gc_layer->set_weight(w, weight-gc_epsilon);
            this->forward(in);
            float_t errorm = this->squared_error(soll);

            gc_layer->set_weight(w, weight);
            this->forward(in);
            this->backward(in, soll);
            float_t gradient = gc_layer->gc_gradient_weights(w);

            float_t q = (errorp-errorm)/(4.0*gc_epsilon);
            float_t d = fabs(q - gradient);
            std::cout <<"FD - dE/d["<< w <<"]="<< q <<" - "<< gradient <<" = " << d <<"\n";
            assert( d < 0.000001 );
        }
    }

    void gc_check_bias(const vec_t& in, const vec_t& soll, layer* gc_layer) {
       
        const my_nn::float_t gc_epsilon = 0.00001;
        
        for (int w=0; w<gc_layer->bias().size(); w++) {

            float_t bias = gc_layer->get_bias(w);

            gc_layer->set_bias(w, bias+gc_epsilon);
            this->forward(in);
            float_t errorp = this->squared_error(soll);

            gc_layer->set_bias(w, bias-gc_epsilon);
            this->forward(in);
            float_t errorm = this->squared_error(soll);

            gc_layer->set_bias(w, bias);
            this->forward(in);
            this->backward(in, soll);
            float_t gradient = gc_layer->gc_gradient_bias(w);

            float_t q = (errorp-errorm)/(4.0*gc_epsilon);
            float_t d = fabs(q - gradient);
            std::cout <<"FD - dE/d["<< w <<"]="<< q <<" - "<< gradient <<" = " << d <<"\n";
            assert( d < 0.000001 );
        }
    }
#endif

private:

    layer *first_layer_ = nullptr;
    layer *last_layer_ = nullptr;
};


} /* namespace my_nn */

#endif /* NETWORK_HPP */
