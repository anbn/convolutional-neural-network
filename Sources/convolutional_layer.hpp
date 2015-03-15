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
    
} /* namespace nn */

#endif
