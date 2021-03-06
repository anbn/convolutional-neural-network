#ifndef NETWORK_HPP
#define NETWORK_HPP

#include "utils.hpp"
#include "activation.hpp"

#include "layer.hpp"
#include "fullyconnected_layer.hpp"
#include "convolutional_layer.hpp"
#include "subsampling_layer.hpp"
#include "softmax_layer.hpp"

namespace nn {

class neural_network {

public:
    
    uint_t in_dim() const { assert(first_layer_!=nullptr); return first_layer_->in_dim(); }
    uint_t out_dim() const { assert(last_layer_!=nullptr); return last_layer_->out_dim(); }

    const vec_t& output() const {
        assert(last_layer_!=nullptr);
        return last_layer_->output();
    }

# if TRAINING_MOMENTUM
    void set_learning_rate(float_t learning_rate) {
        assert(first_layer_!=nullptr);

        layer* l = first_layer_;
        do {
            l->set_learning_rate(learning_rate);
            l = l->next_layer();
        } while(l!=nullptr); 
    }

    float_t learning_rate() {
        assert(first_layer_!=nullptr);
        return first_layer_->learning_rate();
    }
#endif
    
    void add_layer(layer* l) {
        assert(l!=nullptr);

        if (last_layer_==nullptr) {
            first_layer_ = l;
        } else {
            last_layer_->set_next_layer(l);
        }
        last_layer_ = l;
    }

    float_t error(const vec_t& soll) {
        
        output_layer *last = dynamic_cast<output_layer*>(last_layer_);
        assert(last != nullptr);
        
        return last->error(soll);
    }

    void forward(const vec_t& in) {
        assert(first_layer_!=nullptr);

        const vec_t* input = &in;

        layer* l = first_layer_;
        do {
            l->forward(*input);
            input = &l->output();
            l = l->next_layer();
        } while(l!=nullptr); 
    }

    void backward(const vec_t& in, const vec_t& soll) {
       
        output_layer *last = dynamic_cast<output_layer*>(last_layer_);
        assert(last != nullptr);

        --batch_counter_;
        
        last->set_soll(&soll);
        
        layer* l = last_layer_;
        do {
            l->backward(l->prev_layer()==nullptr? in : l->prev_layer()->output(), batch_counter_==0);
            l = l->prev_layer();
        } while(l!=nullptr);

        if ( batch_counter_ == 0 ) {
            batch_counter_ = BATCH_SIZE;
        }
    }

# if GRADIENT_CHECK
    void gc_check_weights(const vec_t& in, const vec_t& soll, layer* gc_layer) {
       
        const nn::float_t gc_epsilon = 0.00001;
        
        for (int w=0; w<gc_layer->weights().size(); w++) {

            float_t weight = gc_layer->get_weight(w);

            gc_layer->set_weight(w, weight+gc_epsilon);
            this->forward(in);
            float_t errorp = this->error(soll);

            gc_layer->set_weight(w, weight-gc_epsilon);
            this->forward(in);
            float_t errorm = this->error(soll);

            gc_layer->set_weight(w, weight);
            this->forward(in);
            this->backward(in, soll);
            float_t gradient = gc_layer->gc_gradient_weights(w);

            float_t q = (errorp-errorm)/(2.0*gc_epsilon);
            float_t d = fabs(q - gradient);
            std::cout <<"FD - dE/d["<< w <<"]="<< q <<" - "<< gradient <<" = " << d <<"\n";
            assert( d < 0.000001 );
        }
    }

    void gc_check_bias(const vec_t& in, const vec_t& soll, layer* gc_layer) {
       
        const nn::float_t gc_epsilon = 0.00001;
        
        for (int w=0; w<gc_layer->bias().size(); w++) {

            float_t bias = gc_layer->get_bias(w);

            gc_layer->set_bias(w, bias+gc_epsilon);
            this->forward(in);
            float_t errorp = this->error(soll);

            gc_layer->set_bias(w, bias-gc_epsilon);
            this->forward(in);
            float_t errorm = this->error(soll);

            gc_layer->set_bias(w, bias);
            this->forward(in);
            this->backward(in, soll);
            float_t gradient = gc_layer->gc_gradient_bias(w);

            float_t q = (errorp-errorm)/(2.0*gc_epsilon);
            float_t d = fabs(q - gradient);
            std::cout <<"FD - dE/d["<< w <<"]="<< q <<" - "<< gradient <<" = " << d <<"\n";
            assert( d < 0.000001 );
        }
    }

    void gc_check() {
    
    }
    
#endif

private:

    layer *first_layer_ = nullptr;
    layer *last_layer_ = nullptr;

    uint batch_counter_ = BATCH_SIZE;
};


} /* namespace nn */

#endif /* NETWORK_HPP */
