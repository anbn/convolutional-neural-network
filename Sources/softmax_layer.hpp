#ifndef SOFTMAX_LAYER_HPP
#define SOFTMAX_LAYER_HPP

namespace nn {
    
class softmax_layer : public output_layer {

public:

    softmax_layer(uint_t in_dim)
        : output_layer(in_dim, in_dim, 0, 0) /* in_dim == out_dim, no weights, no bias */
    {
        std::cout<<"DEBUG: softmax_layer(" <<in_dim<<")\n";
    }


    void forward(const vec_t& in /*[in_dim]*/) {
        assert(in.size() == in_dim_);

        float_t in_max = in[0];
        float_t in_sum = 0.0;
        for (uint_t i=0; i<in_dim_; i++) {
            if (in_max < in[i])
                in_max = in[i];
        }
        for (uint_t o=0; o<out_dim_; o++) {
            output_[o] = exp(in[o] - in_max);
            in_sum += output_[o];
        }
        for (uint_t o=0; o<out_dim_; o++) {
            output_[o] /= in_sum;
        }
    }

    float_t squared_error(const vec_t& soll) const {
        assert(soll.size()==out_dim_);
        
        float_t error = 0.0;
        for(uint_t o=0; o<out_dim_; o++)
            error += (soll[o]-output_[o])*(soll[o]-output_[o]);
        return error;
    }

    float_t in_delta_sum(uint_t fm, uint_t ix, uint_t iy) const {
        assert(fm < in_dim_);
        assert(ix==0 && iy==0);
        
        return (output_[fm]-(*soll_)[fm]);
    }

    void backward(const vec_t& in) {
#if VERBOSE
        std::cout<<"(backwardsoftmax) ";
        //for (uint_t o=0; o<out_dim_; o++) {
        //    delta_[o] = (output_[o]-(*soll_)[o]);
        //    
        //}
#endif
    } 
};

} /* namespace nn */

#endif /* SOFTMAX_LAYER_HPP */
