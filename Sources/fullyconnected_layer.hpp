#ifndef FULLYCONNECTED_LAYER_HPP
#define FULLYCONNECTED_LAYER_HPP

namespace nn {
    
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
            output_[o] = ActFunc.f(sum + bias_[o]);
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
                delta_[o] = (output_[o]-(*soll_)[o]) * ActFunc.df(output_[o]); 
            } else {               /* not the last layer */
                delta_[o] = ActFunc.df(output_[o]) * next_layer_->in_delta_sum(o,0,0);
            }
            for (uint_t i=0; i<in_dim_; i++) {
#if GRADIENT_CHECK
                gc_gradient_weights_[o*in_dim_ + i] = delta_[o] * in[i]; 
#else
                uint_t idx = o*in_dim_+i;
                float_t w = weights_[idx];
                weights_[idx] = weights_[idx]
                                - learning_rate_ * delta_[o] * in[i]
                                + momentum_ * mom_weights_[idx]
                                - learning_rate_ * decay_ * weights_[idx];
                mom_weights_[idx] = weights_[idx] - w;
#endif
            }
#if GRADIENT_CHECK
            gc_gradient_bias_[o] = delta_[o] * 1.0;
#else
            float_t b = bias_[o];
            bias_[o] = bias_[o]
                       - learning_rate_ * delta_[o]*1.0
                       + momentum_*mom_bias_[o];
                       - learning_rate_ * decay_ * bias_[o];
            mom_bias_[o] = bias_[o] - b;
#endif
        }
    } 

    void set_soll( const vec_t* soll ) {
        soll_ = soll;
    }

    ActivationFunction ActFunc;

private:
    const vec_t* soll_ = nullptr;
};

} /* namespace nn */

#endif /* FULLYCONNECTED_LAYER_HPP */
