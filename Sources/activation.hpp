#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP


#include "utils.hpp"

namespace my_nn {

class activation {

public:
    virtual float_t f(float_t x) const = 0;
    virtual float_t df(float_t x) const = 0;
};


class sigmoid : activation {

public:
    float_t f(float_t x) const { return (float_t)1.0 / ((float_t)1.0 + exp(-x)); }
    float_t df(float_t f_x) const { return f_x * (1.0 - f_x); }
};


class tan_h : activation {

public:
    float_t f(float_t x) const {
        const float_t ep = exp(2*x);
        return (ep - 1) / (ep + 1);
    }
    float_t df(float_t f_x) const { return (float_t) 1.0 - f_x * f_x; }
};


class rel : activation {

public:
    float_t f(float_t x) const { return std::max((float_t) 0.0, x); }
    float_t df(float_t f_x) const { return f_x > 0.0 ? 1.0 : 0.0; }
};

} /* namespace my_nn */

#endif /* activation.hpp */

