#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


typedef unsigned char intensity_t;

template <typename T = intensity_t>
class Image {
    
public:

    Image() : width_(0), height_(0) {}
    Image(size_t width, size_t height) : width_(width), height_(height), data_(width * height, 0) {}
   
    template <typename AccessIterator>
    Image(size_t width, size_t height, AccessIterator iter_begin, AccessIterator iter_end)
        : width_(width), height_(height), data_(width * height, 0) {

        std::copy(iter_begin, iter_end, std::begin(data_));
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    std::vector<T>& data() { return data_; }

    void resize(size_t width, size_t height) {
        data_.resize(width * height);
        width_ = width;
        height_ = height;
    }

    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    T& at(size_t x, size_t y) {
        assert(x < width_);
        assert(y < height_);
        return data_[y * width_ + x];
    }

    Image<float> convolution(size_t kernel_size, std::vector<float> kernel ) {
        
        Image<float> result(width_, height_);
        float sum;

        assert( (kernel_size%2==1) && kernel.size()==(kernel_size*kernel_size));

        int s = (kernel_size-1)/2;

        for (int h=s; h<height_-s; h++) {
            for (int w=s; w<width_-s; w++) {
                sum = 0;
                for (int i=-s; i<s+1; i++) {
                    for (int j=-s; j<s+1; j++) {
                        sum += data_[(h+i) * width_ + (w+j)] * kernel[((i+s)*kernel_size)+(j+s)];
                    }
                }
                result.data()[h*width_+w]= sum;
            }
        }
            
        return result;
    }
    

    template <typename CombineFunction>
    Image<T> combinePixels(Image<T> img, CombineFunction f ) {
        
        Image<T> result(width_, height_);

        assert(width_==img.width_ && height_==img.height_);
        
        for (int w=0; w<width_; w++) {
            for (int h=0; h<height_; h++) {
                result.data()[h * width_ + w] = f( this->data_[h * width_ + w], img.data_[h * width_ + w]);
            }
        }
        return result;    
    }
    

    Image<intensity_t> toIntensity(T min, T max) {
        
        Image<intensity_t> result(width_, height_);

        for (int w=0; w<width_; w++) {
            for (int h=0; h<height_; h++) {
                result.data()[h*width_+w] = static_cast<intensity_t>((data_[h * width_ + w]-min)/(max-min)*255.0);
            }
        }

        return result;
    }

    Image<intensity_t> toIntensity() {
        auto pair_minmax = std::minmax_element(std::begin(this->data_), std::end(this->data_));
        return toIntensity(*pair_minmax.first, *pair_minmax.second);
    }


    void importMat(cv::Mat img) {
        this->resize(img.cols, img.rows);
        for (int w=0; w<img.cols; w++) {
            for (int h=0; h<img.rows; h++) {
                data_[h * width_ + w] =
                     0.114 * img.at<cv::Vec3b>(h, w)[0] +
                     0.587 * img.at<cv::Vec3b>(h, w)[1] +
                     0.299 * img.at<cv::Vec3b>(h, w)[2];
            }
        }
    }
    
    cv::Mat exportMat() {
        cv::Mat img(cv::Size(this->width_, this->height_),CV_8UC1);
        for (int w=0; w<img.cols; w++) {
            for (int h=0; h<img.rows; h++) {
                img.at<unsigned char>(h, w, 0) = data_[h * width_ + w];
            }
        }
        return img;
    }
    
private:

    size_t width_;
    size_t height_;
    std::vector<T> data_;

};


Image<intensity_t>
readPgmP5(std::string filename)
{
    Image<intensity_t> result;
    std::ifstream infile;
    std::string s;
    int width, height;
    char a;
    
    infile.open(filename, std::ios::binary | std::ios::in);
    if(!infile.is_open())
        throw "Error opening file.";

    std::getline(infile, s);
    if (s.compare("P5"))
        throw "P5 Format required";

    std::getline(infile, s);
    width = std::stoi(s.substr(0, s.find(" ")));
    height = std::stoi(s.substr(s.find(" ")));
    result.resize(width,height);
    
    std::getline(infile, s);

    for (int h=0; h<height; h++) {
        for (int w=0; w<width; w++) {
            infile.read(&a, 1);
             result.data()[h*width+w] = a;
        }
    }
    infile.close();
    return result;
}

#endif /* image.hpp */
