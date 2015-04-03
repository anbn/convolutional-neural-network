#ifndef FULLYCONNECTED_LAYER_HPP
#define FULLYCONNECTED_LAYER_HPP

namespace nn {
    
template <typename ActivationFunction = sigmoid>
class fullyconnected_layer : public output_layer {

public:

    fullyconnected_layer(uint_t in_width, uint_t out_dim, uint_t in_feature_maps)
        : output_layer(in_feature_maps*in_width*in_width, out_dim, out_dim, out_dim*in_feature_maps*in_width*in_width),
          in_feature_maps_(in_feature_maps),
          in_width_(in_width)
    {
        std::cout<<"DEBUG: fullyconnected_layer(" <<in_width<<","<<out_dim<<","<<in_feature_maps<<")\n";
    }

    fullyconnected_layer(uint_t in_len, uint_t out_dim)
        : fullyconnected_layer(1, out_dim, in_len)
    {}

    void forward(const vec_t& in) {
        assert(in.size() == in_dim_);

        for (uint_t o=0; o<out_dim_; o++) {
            
            float_t sum=0.0;
            for (uint_t ix=0; ix<in_width_; ix++) {
                for (uint_t iy=0; iy<in_width_; iy++) {
                    for (uint_t in_fm=0; in_fm<in_feature_maps_; in_fm++) {
                        sum += in[(in_fm*in_width_ + ix)*in_width_ + iy]
                                * weights_[((o*in_feature_maps_ + in_fm)*in_width_ + ix)*in_width_ + iy];
                    }
                }
            }
            output_[o] = ActFunc.f(sum + bias_[o]);
        }
    }

    float_t error(const vec_t& soll) const {
        assert(soll.size()==out_dim_);
        
        float_t error = 0.0;
        for(uint_t o=0; o<out_dim_; o++)
            error += (soll[o]-output_[o])*(soll[o]-output_[o]);
        return error / 2.0;
    }

    float_t in_delta_sum(uint_t in_fm, uint_t ix, uint_t iy) const {
        assert(in_fm < in_feature_maps_);
        assert(ix < in_width_);
        assert(iy < in_width_);

        float_t sum = 0.0;
        for(uint_t o=0; o<out_dim_; o++)
            sum += delta_[o] * weights_[((o*in_feature_maps_ + in_fm)*in_width_ + ix)*in_width_ + iy];
        return sum;
    }

    /* backpropagation for any layer */
    void backward(const vec_t& in, bool is_update) {
#if VERBOSE
        std::cout<<"(backwardfully) ";
#endif
        assert(in.size()==in_dim_);

        for (uint_t o=0; o<out_dim_; o++) {

            if (next_layer_ == nullptr) {
                /* is the last layer */
                delta_[o] = ActFunc.df(output_[o]) * (output_[o]-(*soll_)[o]);
            } else {
                /* not the last layer, but followed by 1d layer */
                delta_[o] = ActFunc.df(output_[o]) * next_layer_->in_delta_sum(o,0,0);
            }

            /* ordering loops this way saves many comparisons if in_width_==1 */
            for (uint_t ix=0; ix<in_width_; ix++) {
                for (uint_t iy=0; iy<in_width_; iy++) {
                    for (uint_t in_fm=0; in_fm<in_feature_maps_; in_fm++) {
                        uint_t idx = ((o*in_feature_maps_ + in_fm)*in_width_ + ix)*in_width_ + iy;
                        batch_gradient_weights_[idx] += delta_[o] * in[(in_fm*in_width_ + ix)*in_width_ + iy];

                        if ( is_update ) {
                            batch_gradient_weights_[o] = batch_gradient_weights_[o]/BATCH_SIZE;
#if GRADIENT_CHECK
                            gc_gradient_weights_[idx] = batch_gradient_weights_[idx]; 
#elif TRAINING_MOMENTUM
                            float_t w = weights_[idx];
                            weights_[idx] = weights_[idx]
                                - learning_rate_ * batch_gradient_weights_[idx]
                                + momentum_ * mom_weights_[idx]
                                - learning_rate_ * decay_ * weights_[idx];
                            mom_weights_[idx] = weights_[idx] - w;
#elif TRAINING_ADADELTA
                            ad_acc_gradient_weights_[idx] = ad_ro_*ad_acc_gradient_weights_[idx] + (1.0-ad_ro_)*batch_gradient_weights_[idx]*batch_gradient_weights_[idx];
                            float_t ad_update = - sqrt((ad_acc_updates_weights_[idx]+ad_epsilon_)/(ad_acc_gradient_weights_[idx]+ad_epsilon_)) * batch_gradient_weights_[idx];
                            ad_acc_updates_weights_[idx] = ad_ro_*ad_acc_updates_weights_[idx] + (1.0-ad_ro_)*ad_update*ad_update;
                            weights_[idx] = weights_[idx]+ad_update;
#endif
                            batch_gradient_weights_[idx] = 0.0;
                        } 
                    }
                }
            }
            batch_gradient_bias_[o] += delta_[o] * 1.0;

            if ( is_update ) {
                batch_gradient_bias_[o] = batch_gradient_bias_[o]/BATCH_SIZE;
#if GRADIENT_CHECK
                gc_gradient_bias_[o] = batch_gradient_bias_[o];
#elif TRAINING_MOMENTUM
                float_t b = bias_[o];
                bias_[o] = bias_[o]
                    - learning_rate_ * batch_gradient_bias_[o]
                    + momentum_*mom_bias_[o]
                    - learning_rate_ * decay_ * bias_[o];
                mom_bias_[o] = bias_[o] - b;
#elif TRAINING_ADADELTA
                ad_acc_gradient_bias_[o] = ad_ro_*ad_acc_gradient_bias_[o] + (1.0-ad_ro_)*batch_gradient_bias_[o]*batch_gradient_bias_[o];
                float_t ad_update = - sqrt((ad_acc_updates_bias_[o]+ad_epsilon_)/(ad_acc_gradient_bias_[o]+ad_epsilon_)) * batch_gradient_bias_[o];
                ad_acc_updates_bias_[o] = ad_ro_*ad_acc_updates_bias_[o] + (1.0-ad_ro_)*ad_update*ad_update;
                bias_[o] = bias_[o]+ad_update;
#endif
                batch_gradient_bias_[o] = 0.0;
            }
        }
    }

    ActivationFunction ActFunc;
    uint_t in_feature_maps_;
    uint_t in_width_;
};

} /* namespace nn */

#endif /* FULLYCONNECTED_LAYER_HPP */
