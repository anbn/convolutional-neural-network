#ifndef SUBSAMPLING_LAYER_HPP
#define SUBSAMPLING_LAYER_HPP

namespace nn {

template <typename ActivationFunction = sigmoid>
class subsampling_layer : public layer {

public:

    subsampling_layer(uint_t in_width, uint_t out_width, uint_t feature_maps, uint_t block_size)
            : layer(feature_maps*in_width*in_width, feature_maps*out_width*out_width, feature_maps, feature_maps),
            feature_maps_(feature_maps),
            block_size_(block_size),
            in_width_(in_width), out_width_(out_width)
    {

        std::cout<<"DEBUG: subsampling_layer(" <<in_width_<<","<<out_width_<<","<<feature_maps_<<","<<block_size_<<")\n";
        
        assert(in_width == out_width*block_size);
    }

    void forward(const vec_t& in /*[in_feature_map * in_width_ * in_width_]*/) {
        
        assert(in.size() == feature_maps_ * in_width_ * in_width_);

        for (uint_t fm=0; fm<feature_maps_; fm++) {
            
            for (uint_t ox=0; ox<out_width_; ox++) {
                for (uint_t oy=0; oy<out_width_; oy++) {

#if POOLING_AVG
                    float_t sum_block = 0.0;
                    for (uint_t bx=0; bx<block_size_; bx++) {
                        for (uint_t by=0; by<block_size_; by++) {
                            sum_block += in[(fm*in_width_ + (block_size_*ox+bx))*in_width_ + (block_size_*oy+by)];
                        }
                    }
#elif POOLING_MAX
                    float_t sum_block = in[(fm*in_width_ + (block_size_*ox+0))*in_width_ + (block_size_*oy+0)];
                    for (uint_t bx=0; bx<block_size_; bx++) {
                        for (uint_t by=0; by<block_size_; by++) {
                            sum_block = std::max(sum_block, in[(fm*in_width_ + (block_size_*ox+bx))*in_width_ + (block_size_*oy+by)]);
                        }
                    }
#else
                    std::cout<<"Error: no pooling method defined.\n";
                    exit(1);
#endif
                    output_[(fm*out_width_ + ox)*out_width_ + oy] = A_.f( weights_[fm]*sum_block + bias_[fm] );
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
    
    void backward(const vec_t& in, bool is_update) {
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
#if POOLING_AVG
                    float_t sum_block = 0.0;
                    for (uint_t bx=0; bx<block_size_; bx++) {
                        for (uint_t by=0; by<block_size_; by++) {
                            sum_block += in[(fm*in_width_ + (block_size_*ox+bx  ))*in_width_ + (block_size_*oy+by  )];
                        }
                    }
#elif POOLING_MAX
                    float_t sum_block = in[(fm*in_width_ + (block_size_*ox+0  ))*in_width_ + (block_size_*oy+0  )];
                    for (uint_t bx=0; bx<block_size_; bx++) {
                        for (uint_t by=0; by<block_size_; by++) {
                            sum_block = std::max(sum_block, in[(fm*in_width_ + (block_size_*ox+bx  ))*in_width_ + (block_size_*oy+by  )]);
                        }
                    }
#else
                    std::cout<<"Error: no pooling method defined.\n";
                    exit(1);
#endif
                    sum += sum_block * delta_[out_index];
                }
            }

            batch_gradient_weights_[fm] += sum;
            batch_gradient_bias_[fm] += sum_delta;

            if ( is_update ) {
#if GRADIENT_CHECK
                gc_gradient_weights_[fm] = sum;
                gc_gradient_bias_[fm] = sum_delta;
#elif TRAINING_MOMENTUM
                float_t w = weights_[fm];
                weights_[fm] = weights_[fm]
                    - learning_rate_*sum
                    + momentum_ * mom_weights_[fm]
                    - learning_rate_ * decay_ * weights_[fm];
                mom_weights_[fm] = weights_[fm] - w;

                float_t b = bias_[fm];
                bias_[fm] = bias_[fm]
                    - learning_rate_ * sum_delta
                    + momentum_ * mom_bias_[fm]
                    - learning_rate_ * decay_ * bias_[fm];
                mom_bias_[fm] = bias_[fm] - b;
#elif TRAINING_ADADELTA
                ad_acc_gradient_weights_[fm] = ad_ro_*ad_acc_gradient_weights_[fm] + (1-ad_ro_)*sum*sum;
                float_t ad_update = - sqrt((ad_acc_updates_weights_[fm]+ad_epsilon_)/(ad_acc_gradient_weights_[fm]+ad_epsilon_)) * sum;
                ad_acc_updates_weights_[fm] = ad_ro_*ad_acc_updates_weights_[fm] + (1-ad_ro_)*ad_update*ad_update;
                weights_[fm] = weights_[fm]+ad_update;

                ad_acc_gradient_bias_[fm] = ad_ro_*ad_acc_gradient_bias_[fm] + (1-ad_ro_)*sum_delta*sum_delta;
                ad_update = - sqrt((ad_acc_updates_bias_[fm]+ad_epsilon_)/(ad_acc_gradient_bias_[fm]+ad_epsilon_)) * sum_delta;
                ad_acc_updates_bias_[fm] = ad_ro_*ad_acc_updates_bias_[fm] + (1-ad_ro_)*ad_update*ad_update;
                bias_[fm] = bias_[fm]+ad_update;
#endif
                batch_gradient_weights_[fm] = 0.0;
                batch_gradient_bias_[fm] = 0.0;

            }
        }
    }

private:
    
    ActivationFunction A_;

    uint_t feature_maps_;   /* feature_maps_ == in_feature_maps_ == out_feature_maps_ */
    uint_t block_size_;
    uint_t in_width_, out_width_;
};

} /* namespace nn */

#endif /* SUBSAMPLING_LAYER_HPP */
