#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include "utils.hpp"
#include "image.hpp"

namespace nn {

class mnist_reader {
    
public:

    uint_t num_examples() const { return num_examples_; }
    uint_t image_width() const { return image_width_; }
    uint_t image_height() const { return image_height_; }

    const Image<float_t>& image(uint_t n) { return images_[n]; }
    int   label(uint_t n) { return labels_[n]; }

    mnist_reader() : num_examples_(0) {}


    bool read(std::string path_images, std::string path_labels, uint_t num) {
        std::cout<<"Reading mnist... ";
        
        if (num == 0 || num > 60000) {
            num = 60000;
        }

        images_.resize(num);
        labels_.resize(num);

        std::ifstream file_images(path_images, std::ios::in|std::ios::binary);
        std::ifstream file_labels(path_labels, std::ios::in|std::ios::binary);

        if (!file_images.is_open() || !file_labels.is_open()) {
            std::cout<<"\nError reading mnist files ("<<path_images<<", "<<path_labels<<")\n";
            file_images.close();
            file_labels.close();
            return false;
        }

        uint32_t magic_images, magic_labels, num_images, num_labels, rows, columns;
        
        file_images.read(reinterpret_cast<char*>(&magic_images), sizeof(uint32_t));
        file_labels.read(reinterpret_cast<char*>(&magic_labels), sizeof(uint32_t));
        endswap(&magic_images);
        endswap(&magic_labels);
        if (magic_images != 0x0803 || magic_labels != 0x0801) {
            std::cout<< "\nUnexpected magic number, probably not a MNIST file.\n";
            file_images.close();
            file_labels.close();
            return false;
        }

        file_images.read(reinterpret_cast<char*>(&num_images), sizeof(uint32_t));
        file_labels.read(reinterpret_cast<char*>(&num_labels), sizeof(uint32_t));
        endswap(&num_images);
        endswap(&num_labels);

        file_images.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        file_images.read(reinterpret_cast<char*>(&columns), sizeof(uint32_t));
        endswap(&rows);
        endswap(&columns);

        unsigned char byte, label;
        std::vector<float_t> data(rows*columns);
        for (uint_t n = 0; n<num; ++n) {
            file_labels.read(reinterpret_cast<char*>(&label), sizeof(unsigned char));
            for (uint_t i=0; i<rows*columns; i++) {
                file_images.read(reinterpret_cast<char*>(&byte), sizeof(unsigned char));
                data[i] = (byte / 128.0)-1; /* scale to [-1,1] */
            }
            images_[n] = Image<float_t>(columns, rows, std::begin(data), std::end(data));
            labels_[n] = label;
            ++num_examples_;
        }

        file_images.close();
        file_labels.close();

        std::cout<<"done. ("<<num_examples_<<" images read)\n";
        return true;
    }

private:
    
    uint_t num_examples_;
        
    uint_t image_width_;
    uint_t image_height_;
    std::vector< Image<float_t> > images_;
    std::vector<int> labels_;
};




void mnist_reader_test() {
    mnist_reader mnist;
    mnist.read("data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 15);

    for (int i=0; i<mnist.num_examples(); i++) {
        Image<nn::float_t> img = mnist.image(i);
        cv::imshow("mnist["+std::to_string(i)+"]: "+std::to_string(mnist.label(i)), img.toIntensity(-1,1).exportMat());
        while(cv::waitKey(0)!=27);
    }
}



} /* namespace nn */

#endif /* MNIST_READER.HPP */

