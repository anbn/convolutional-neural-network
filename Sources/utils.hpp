#ifndef UTILS_HPP
#define UTILS_HPP

#include <random>

namespace nn {

typedef unsigned int uint_t;
typedef double float_t;
typedef std::vector<float_t> vec_t;

template<typename T>
inline T get_random(T min, T max) {
    static std::mt19937 mt_rand((0));
    std::uniform_real_distribution<T> dst(min, max);
    return dst(mt_rand);
}

template<typename Iter>
void get_random(Iter begin, Iter end, float_t min, float_t max) {
    for (Iter it = begin; it != end; ++it) 
        *it = get_random(min, max);
}

template <typename T>
void endswap(T *obj)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(obj);
  std::reverse(memp, memp + sizeof(T));
}

} /* namespace nn */

#endif /* UTILS_HPP */
