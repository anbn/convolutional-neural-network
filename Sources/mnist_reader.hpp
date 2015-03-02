#ifndef MNIST_READER_HPP
#define MNIST_READER_HPP

#include "utils.hpp"
#include "image.hpp"

namespace my_nn {

typedef double float_t;
typedef std::vector<float_t> vec_t;

class mnist_reader {
    
public:

    size_t num_examples() const { return num_examples_; }
    size_t image_width() const { return image_width_; }
    size_t image_height() const { return image_height_; }

    const Image<float_t>& image(int n) { return images_[n]; }
    int   label(int n) { return labels_[n]; }

    mnist_reader() : num_examples_(0) {};

//    void rescale(float_t min, float_t max) {
//        for (auto& img : images_)
//            img.rescale(min, max);
//    
//    }
    
    bool read(std::string filename_images, std::string filename_labels, size_t num) {
        std::cout<<"Reading mnist... ";
        
        if (num == 0 || num > 60000) {
            std::cout<<"\nWarning: setting number of images to maximum of 60000.\n";
            num = 60000;
        }

        images_.resize(num);
        labels_.resize(num);

        std::ifstream file_images(filename_images, std::ios::in|std::ios::binary);
        std::ifstream file_labels(filename_labels, std::ios::in|std::ios::binary);

        if (!file_images.is_open() || !file_labels.is_open()) {
            std::cout<<"\nError reading mnist files ("<<filename_images<<", "<<filename_labels<<")\n";
            file_images.close();
            file_labels.close();
            return false;
        }

        uint32_t magic_images, magic_labels, num_images, num_labels, rows, columns;
        
        file_images.read(reinterpret_cast<char*>(&magic_images), sizeof(uint32_t));
        file_labels.read(reinterpret_cast<char*>(&magic_labels), sizeof(uint32_t));
        endswap(&magic_images);
        endswap(&magic_labels);
        if (magic_images != 0x00000803 || magic_labels != 0x00000801) {
            std::cout<< "\nUnexpected magic numbers, probably not a MNIST file.\n";
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
        for (size_t n = 0; n<num; ++n) {
            
            file_labels.read(reinterpret_cast<char*>(&label), sizeof(unsigned char));
            for (int i=0; i<rows*columns; i++) {
                file_images.read(reinterpret_cast<char*>(&byte), sizeof(unsigned char));
                data[i] = (byte / 128.0)-1;
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

    void corrupt(int n) 
    {
        images_[n].fill(0);
    }
protected:

    size_t num_examples_;
        
    size_t image_width_;
    size_t image_height_;
    std::vector< Image<float_t> > images_;
    std::vector<int> labels_;
};


} /* namespace my_nn */

#endif /* MNIST_READER.HPP */

