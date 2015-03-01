#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include "utils.hpp"
#include "activation.hpp"
#include "image.hpp"

namespace my_nn {

typedef double float_t;
typedef std::vector<float_t> vec_t;

class mnist_reader {
    
public:

    size_t num_examples() const { return num_examples_; }
    size_t image_width() const { return image_width_; }
    size_t image_height() const { return image_height_; }

    vec_t& image(int n) { return images_[n]; }
    int    label(int n) { return labels_[n]; }

    mnist_reader(std::string file_images, std::string file_labels) {

    }

protected:
        
    size_t num_examples_;
    size_t image_width_;
    size_t image_height_;
    std::vector<vec_t> images_;
    std::vector<int> labels_;
};


} /* namespace my_nn */

#endif /* MNIST_READER.HPP */

