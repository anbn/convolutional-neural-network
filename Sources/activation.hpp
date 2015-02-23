#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

class activation {

public:
    virtual float f(float x) const = 0;
    virtual float df(float x) const = 0;
};


class sigmoid : activation {

public:
    float f(float x) const { return 1.0 / (1.0 + exp(-x)); }
    float df(float f_x) const { return f_x * (1.0 - f_x); }
};

 
class rel : activation {

public:
    float_t f(float_t x) const { return std::max((float_t) 0.0, x); }
    float_t df(float_t f_x) const { return f_x > 0.0 ? 1.0 : 0.0; }
};

#endif /* activation.hpp */

