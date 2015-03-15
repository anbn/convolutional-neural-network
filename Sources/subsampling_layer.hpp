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

} /* namespace nn */

#endif /* SUBSAMPLING_LAYER_HPP */
