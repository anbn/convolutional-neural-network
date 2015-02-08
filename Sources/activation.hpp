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


#endif /* activation.hpp */

