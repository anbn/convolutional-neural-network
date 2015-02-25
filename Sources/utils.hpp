#ifndef UTILS_HPP
#define UTILS_HPP

#include <random>

template<typename T>
inline T uniform_rand(T min, T max) {
    static std::mt19937 mt_rand((0));
    std::uniform_real_distribution<T> dst(min, max);
    return dst(mt_rand);
}

template<typename Iter>
void uniform_rand(Iter begin, Iter end, float_t min, float_t max) {
    for (Iter it = begin; it != end; ++it) 
        *it = uniform_rand(min, max);
}

#endif /* UTILS_HPP */
