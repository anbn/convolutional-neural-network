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

    size_t in_dim() const { return in_dim_; }
    size_t out_dim() const { return out_dim_; }

    const vec_t& output() const { return output_; }
    const vec_t& delta() const { return delta_; }
    const vec_t& weights() const { return weights_; }
    const vec_t& bias() const { return bias_; }

    void resetWeights() {
        randomize(std::begin(weights_), std::end(weights_), -.5, .5);
        randomize(std::begin(bias_), std::end(bias_), -0.1, 0.1);
    }

    void set_learningrate(float_t l) {
        learning_rate = l;
    }

    virtual float_t in_delta_sum(int fm, int ix,int iy) const = 0;
    virtual void forward(const vec_t& in /*[in_dim]*/) = 0;

#if GRADIENT_CHECK
    vec_t gc_gradient_weights_,
          gc_gradient_bias_;

    float_t gc_gradient_weights(int i) const { return gc_gradient_weights_[i]; }
    float_t gc_gradient_bias(int i) const { return gc_gradient_bias_[i]; }
    float_t get_weight(int i) const { return weights_[i]; }
    void    set_weight(int i, float_t v) { weights_[i] = v; }
    float_t get_bias(int i) const { return bias_[i]; }
    void    set_bias(int i, float_t v) { bias_[i] = v; }
#endif

protected:
    
    layer(size_t in_dim, size_t out_dim, size_t bias_dim, size_t weights_dim) : in_dim_(in_dim), out_dim_(out_dim) {
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

    
    size_t in_dim_;
    size_t out_dim_;

    vec_t weights_;     /* variable size, depending on layer type */
    vec_t delta_;       /* [out_dim_] */
    vec_t bias_;        /* [feature_map] */

    vec_t output_;      /* [feature_maps * out_dim_ * out_dim_] */

    float_t learning_rate = 0.0001;
};

//------------------------------------------------------------------------------

template <typename ActivationFunction = sigmoid>
class fullyconnected_layer : public layer {

public:

    fullyconnected_layer(size_t in_dim, size_t out_dim)
        : layer(in_dim, out_dim, out_dim, in_dim*out_dim) {
        std::cout<<"DEBUG: fullyconnected_layer(" <<in_dim<<","<<out_dim<<")\n";
    };


    void forward(const vec_t& in /*[in_dim]*/) {
        
        assert(in.size() == in_dim_);

        for (int o=0; o<out_dim_; o++) {
            
            float_t sum=0.0;
            for (int i=0; i<in_dim_; i++) {
                sum += in[i] * weights_[o*in_dim_ + i];
            }
            output_[o] = A_.f(sum + bias_[o]);
        }
    }

    float_t squared_error(const vec_t& soll) {

        assert(soll.size()==out_dim_);
        
        float_t error = 0.0;
        for(int o=0; o<out_dim_; o++)
            error += (soll[o]-output_[o])*(soll[o]-output_[o]);
        return error;
    }

    float_t in_delta_sum(int fm, int ix, int iy) const {
        
        assert(fm < in_dim_);
        assert(ix==0 && iy==0);
        
        float_t sum = 0.0;
        for(int o=0; o<out_dim_; o++)
            sum += delta_[o] * weights_[o*in_dim_ + fm];
        return sum;
    }

    /* backpropagation for any layer except the last one */
    void backward(const vec_t& in, const fullyconnected_layer& next_layer) {
#if VERBOSE
        std::cout<<"(backwardfully) ";
#endif

        assert(in.size()==in_dim_);
        assert(next_layer.in_dim()==out_dim_);

        for (int o=0; o<out_dim_; o++) {

            delta_[o] = A_.df(output_[o]) * next_layer.in_delta_sum(o,0,0); //vs. delta_[o] = output_[o] * A_.df(output_[o]) * sum; // check at home
           
            for (int i=0; i<in_dim_; i++) {
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

    
    /* backpropagation for the last layer */
    void backward(const layer& previous_layer, const vec_t& soll) {
#if VERBOSE
        std::cout<<"(backwardfullylast) ";
#endif
        assert(previous_layer.out_dim()==in_dim_);
        assert(soll.size()==out_dim_);

        for (int o=0; o<out_dim_; o++) {

            delta_[o] = (output_[o]-soll[o]) * A_.df(output_[o]);

            for (int i=0; i<in_dim_; i++) {
#if GRADIENT_CHECK
                gc_gradient_weights_[o*in_dim_ + i] = delta_[o]*previous_layer.output()[i]; 
#else
                weights_[o*in_dim_ + i] -= learning_rate * delta_[o] * previous_layer.output()[i];
#endif
            }
#if GRADIENT_CHECK
            gc_gradient_bias_[o] = delta_[o] * 1.0;
#else
            bias_[o] -= learning_rate * delta_[o] * 1.0;
#endif
        } 
    }

    ActivationFunction A_;
};


/* A: ActivationFunction */
template <typename ActivationFunction = sigmoid>
class convolutional_layer : public layer {

    class connection_table {

    public:
        connection_table() : in_feature_maps_(0), out_feature_maps_(0) {};
        connection_table(const bool *p, size_t in, size_t out)
        : connected_(in*out),
        in_feature_maps_(in),
        out_feature_maps_(out) {
            std::copy(p, p + in*out, connected_.begin());
        };

        inline bool is_connected(int i, int o) const {
            return (in_feature_maps_==0) && (out_feature_maps_==0) ? true : connected_[i * out_feature_maps_ + o];
        }

    private:
        std::vector<bool> connected_;
        size_t in_feature_maps_;
        size_t out_feature_maps_;
    };

    
public:

    convolutional_layer(size_t in_width, size_t out_width, size_t in_feature_maps, size_t out_feature_maps)
            : layer(in_feature_maps*in_width*in_width, out_feature_maps*out_width*out_width, out_feature_maps, in_feature_maps*out_feature_maps*(in_width-out_width+1)*(in_width-out_width+1)),
            in_feature_maps_(in_feature_maps),out_feature_maps_(out_feature_maps),
            in_width_(in_width), out_width_(out_width),
            filter_width_(in_width-out_width+1) {
        std::cout<<"DEBUG: convolutional_layer(" <<in_width<<","<<out_width<<","<<in_feature_maps<<","<<out_feature_maps_<<")\n";
    }

    size_t out_feature_maps() const { return out_feature_maps_; }
    size_t in_feature_maps() const { return in_feature_maps_; }
    size_t filter_width() const { return filter_width_; }

    void set_connection(const bool *p) {
        connection_ = connection_table(p, in_feature_maps_, out_feature_maps_);
    }


    void forward(const vec_t& in /*[in_feature_map * in_width * in_width]*/) {
        
        assert(in.size() == in_feature_maps_ * in_width_ * in_width_);
        assert(weights_.size() == in_feature_maps_*out_feature_maps_*filter_width_*filter_width_);

        
        for (int out_fm=0; out_fm<out_feature_maps_; out_fm++) {

            for (int ox=0; ox<out_width_; ox++) {
                for (int oy=0; oy<out_width_; oy++) {

                    float_t sum = 0.0;
                    for (int in_fm=0; in_fm<in_feature_maps_; in_fm++) {

                        if (!connection_.is_connected(in_fm, out_fm))
                            continue;

                        for (int fx=0; fx<filter_width_; fx++) {
                            for (int fy=0; fy<filter_width_; fy++) {
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
    float_t in_delta_sum(int in_fm, int ix, int iy) const {

        assert(in_fm < in_feature_maps_);
        assert(ix < in_width_);
        assert(iy < in_width_);

        // FIX ME
        float_t sum = 0.0;
        for (int out_fm=0; out_fm<out_feature_maps_; out_fm++) {
            
            if(!connection_.is_connected(in_fm, out_fm))
                continue;

            for(int fx = 0; fx<filter_width_; fx++ ) {
                for(int fy = 0; fy<filter_width_; fy++ ) {
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
    void backward(const vec_t& in, const layer& next_layer) {
#if VERBOSE
        std::cout<<"(backwardconv) ";
#endif
        assert(in.size()==in_feature_maps_ * in_width_ * in_width_);
        assert(next_layer.in_dim()==out_dim_);
        assert(weights_.size() == in_feature_maps_*out_feature_maps_*filter_width_*filter_width_);


        for (int out_fm=0; out_fm<out_feature_maps_; out_fm++) {
            
            float_t sum_delta = 0.0;
            for (int ox=0; ox<out_width_; ox++) {
                for (int oy=0; oy<out_width_; oy++) {
                    int out_index = (out_fm*out_width_+ ox)*out_width_ + oy;
                    delta_[out_index] = A_.df(output_[out_index]) * next_layer.in_delta_sum(out_fm, ox, oy);
                    sum_delta += delta_[out_index];
                }
            }
#if GRADIENT_CHECK
            gc_gradient_bias_[out_fm] = sum_delta;
#else
            bias_[out_fm] -= learning_rate * sum_delta;
#endif       
            for (int in_fm=0; in_fm<in_feature_maps_; in_fm++) {

                if (!connection_.is_connected(in_fm, out_fm))
                    continue;

                for (int fx=0; fx<filter_width_; fx++) {
                    for (int fy=0; fy<filter_width_; fy++) {

                        float_t sum = 0.0;
                        for (int ox=0; ox<out_width_; ox++) {
                            for (int oy=0; oy<out_width_; oy++) {
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

    connection_table connection_;
    size_t in_feature_maps_, out_feature_maps_;
    size_t in_width_, out_width_;
    size_t filter_width_;
};


template <typename ActivationFunction = sigmoid>
class subsampling_layer : public layer {

public:

    subsampling_layer(size_t in_width, size_t out_width, size_t feature_maps, size_t block_size)
            : layer(feature_maps*in_width*in_width, feature_maps*out_width*out_width, feature_maps, feature_maps),
            feature_maps_(feature_maps),
            block_size_(block_size),
            in_width_(in_width), out_width_(out_width) {

        std::cout<<"DEBUG: subsampling_layer(" <<in_width_<<","<<out_width_<<","<<feature_maps_<<", "<<block_size_<<")\n";
        
        assert(in_width == out_width*block_size);
    }

    void forward(const vec_t& in /*[in_feature_map * in_width_ * in_width_]*/) {
        
        assert(in_width_ == out_width_*block_size_); // unecessary, fix later
        assert(in.size() == feature_maps_ * in_width_ * in_width_);

        for (int fm=0; fm<feature_maps_; fm++) {
            
            for (int ox=0; ox<out_width_; ox++) {
                for (int oy=0; oy<out_width_; oy++) {

                    float_t sum = 0.0;
                    for (int bx=0; bx<block_size_; bx++) {
                        for (int by=0; by<block_size_; by++) {
                            sum += in[(fm*in_width_ + (block_size_*ox+bx))*in_width_ + (block_size_*oy+by)];
                        }
                    }

                    //float_t sum =   in[(fm*in_width_ + (2*ox  ))*in_width_ + (2*oy  )] +
                    //                in[(fm*in_width_ + (2*ox+1))*in_width_ + (2*oy  )] + 
                    //                in[(fm*in_width_ + (2*ox  ))*in_width_ + (2*oy+1)] + 
                    //                in[(fm*in_width_ + (2*ox+1))*in_width_ + (2*oy+1)];
                   
                    output_[(fm*out_width_ + ox)*out_width_ + oy] = A_.f( weights_[fm]*sum + bias_[fm] );
                    //output_[(fm*out_width_ + ox)*out_width_ + oy] = A_.f(sum*0.25  );
                }
            }
        }
    }
    
    float_t in_delta_sum(int fm, int ix, int iy) const {

        assert(fm < feature_maps_);
        assert(ix < in_width_);
        assert(iy < in_width_);
        
        return delta_[(fm*out_width_ + ix/2)*out_width_ + iy/2] * weights_[fm];
    }
    
    void backward(const vec_t& in, const layer& next_layer) {
#if VERBOSE
        std::cout<<"(backwardsub) ";
#endif
        assert(in.size()==in_dim_);
        assert(next_layer.in_dim()==out_dim_);

        for (int fm=0; fm<feature_maps_; fm++) {

            float_t sum = 0.0;
            float_t sum_delta = 0.0;
            for (int ox=0; ox<out_width_; ox++) {
                for (int oy=0; oy<out_width_; oy++) {

                    const int out_index = (fm*out_width_+ ox)*out_width_ + oy;
                    delta_[out_index] = A_.df(output_[out_index]) * next_layer.in_delta_sum(fm,ox,oy);
                    sum_delta += delta_[out_index]; 
                    
                    float_t sum_block = 0.0;
                    for (int bx=0; bx<block_size_; bx++) {
                        for (int by=0; by<block_size_; by++) {
                            sum_block += in[(fm*in_width_ + (block_size_*ox+bx  ))*in_width_ + (block_size_*oy+by  )];
                        }
                    }
                    sum += sum_block * delta_[out_index];
                    //sum += (in[(fm*in_width_ + (2*ox  ))*in_width_ + (2*oy  )] +
                    //        in[(fm*in_width_ + (2*ox+1))*in_width_ + (2*oy  )] + 
                    //        in[(fm*in_width_ + (2*ox  ))*in_width_ + (2*oy+1)] + 
                    //        in[(fm*in_width_ + (2*ox+1))*in_width_ + (2*oy+1)]) *
                    //        delta_[out_index];
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

    size_t feature_maps_;   /* feature_maps_ == in_feature_maps_ == out_feature_maps_ */
    size_t block_size_;
    size_t in_width_, out_width_;
};


class neural_network {

public:
    
    size_t in_dim() const { assert(!layers_.empty()); return layers_.front()->in_dim(); }
    size_t out_dim() const { assert(!layers_.empty()); return layers_.back()->out_dim(); }

    const vec_t& output() const { assert(!layers_.empty()); return layers_.back()->output(); };

    void set_learningrate(float_t lr) {
        for (auto& l : layers_) {
            l->set_learningrate(lr);
        }
    }
    
    void add_layer(layer* l) {
        assert(layers_.empty() || layers_.back()->out_dim() == l->in_dim());
        layers_.push_back(l);
    }

    void forward(const vec_t& in) {
    
        assert(!layers_.empty());

        const vec_t *input = &in;
        for (auto& l : layers_) {
            l->forward(*input);
            input = &l->output();
        }
    }

    void backward(const vec_t& in, const vec_t& soll) {

    }

private:
    std::list<layer*> layers_;
};


} /* namespace my_nn */

#endif /* NETWORK_HPP */
