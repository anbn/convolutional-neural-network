#ifndef CONVOLUTIONAL_LAYER_HPP
#define CONVOLUTIONAL_LAYER_HPP

namespace nn {

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
        assert(in_width_>=out_width_);

        std::cout<<"DEBUG: convolutional_layer(" <<in_width_<<","<<out_width_<<","<<in_feature_maps_<<","<<out_feature_maps_<<") filter_width:"<<filter_width_<<"\n";

        /* initialize connection table */
        connected_.resize(in_feature_maps_*out_feature_maps);
        std::fill(std::begin(connected_), std::end(connected_), true);
        reset_weights(filter_width_*filter_width_*in_feature_maps_);
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

        if (dropout_prob_ > 0.0)
            sample_dropout();

        for (uint_t out_fm=0; out_fm<out_feature_maps_; out_fm++) {

            for (uint_t ox=0; ox<out_width_; ox++) {
                for (uint_t oy=0; oy<out_width_; oy++) {

                    /* dropout while training: [0,1), dropout while testing: 1 */
                    if (dropout_prob_!= 0 && dropout_prob_!=1 && dropout_sample_[(out_fm*out_width_+ ox)*out_width_ + oy]<dropout_prob_) {
                        output_[(out_fm*out_width_+ ox)*out_width_ + oy] = 0.0;
                        continue;
                    }

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
                    /* dropout while testing */
                    if (dropout_prob_==1)
                        sum = sum*0.5;

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
    void backward(const vec_t& in, bool is_update) {
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

            batch_gradient_bias_[out_fm] += sum_delta;

            if ( is_update ) {
#if GRADIENT_CHECK
                gc_gradient_bias_[out_fm] = batch_gradient_bias_[out_fm] ;
#elif TRAINING_MOMENTUM
                float_t b = bias_[out_fm];
                bias_[out_fm] = bias_[out_fm]
                    - learning_rate_ * batch_gradient_bias_[out_fm] 
                    + momentum_ * mom_bias_[out_fm]
                    - learning_rate_ * decay_ * bias_[out_fm];
                mom_bias_[out_fm] = bias_[out_fm] - b;
#elif TRAINING_ADADELTA
                ad_acc_gradient_bias_[out_fm] = ad_ro_*ad_acc_gradient_bias_[out_fm] + (1-ad_ro_)*batch_gradient_bias_[out_fm]*batch_gradient_bias_[out_fm] ;
                float_t ad_update = - sqrt((ad_acc_updates_bias_[out_fm]+ad_epsilon_)/(ad_acc_gradient_bias_[out_fm]+ad_epsilon_)) * batch_gradient_bias_[out_fm] ;
                ad_acc_updates_bias_[out_fm] = ad_ro_*ad_acc_updates_bias_[out_fm] + (1-ad_ro_)*ad_update*ad_update;
                bias_[out_fm] = bias_[out_fm]+ad_update;
#endif       
                batch_gradient_bias_[out_fm] = 0.0;
            }


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
                        uint_t idx = ((in_fm*out_feature_maps_ + out_fm)*filter_width_ + fx)*filter_width_ + fy;
                        batch_gradient_weights_[idx] += sum;

                        if (is_update) {
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
                            ad_acc_gradient_weights_[idx] = ad_ro_*ad_acc_gradient_weights_[idx] + (1-ad_ro_)*batch_gradient_weights_[idx]*batch_gradient_weights_[idx];
                            float_t ad_update = - sqrt((ad_acc_updates_weights_[idx]+ad_epsilon_)/(ad_acc_gradient_weights_[idx]+ad_epsilon_)) * batch_gradient_weights_[idx];
                            ad_acc_updates_weights_[idx] = ad_ro_*ad_acc_updates_weights_[idx] + (1-ad_ro_)*ad_update*ad_update;
                            weights_[idx] = weights_[idx]+ad_update;
#endif
                            batch_gradient_weights_[idx] = 0.0;
                        }
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
    
} /* namespace nn */

#endif /* CONVOLUTIONAL_LAYER_HPP */
