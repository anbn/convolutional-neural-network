#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <algorithm> /* for std::minmax */

typedef unsigned char intensity_t;
typedef unsigned int uint_t;


template <typename T = intensity_t>
class Image {
    
public:

    Image() : width_(0), height_(0) {}
    Image(uint_t width, uint_t height)
        : width_(width), height_(height),
        data_(width * height, 0)
    {}
   
    Image(uint_t width, uint_t height, const std::vector<T> &data)
        : width_(width), height_(height)
    {
        data_.resize(width_*height_);
        std::copy(std::begin(data), std::end(data), std::begin(data_));
    }
    template <typename AccessIterator>
    Image(uint_t width, uint_t height, AccessIterator iter_begin, AccessIterator iter_end)
        : width_(width), height_(height),
        data_(width * height)
    {
        data_.resize(width_*height_);
        std::copy(iter_begin, iter_end, std::begin(data_));
    }

    Image(cv::Mat img) {
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
    
    uint_t width() const { return width_; }
    uint_t height() const { return height_; }
    const std::vector<T>& data() const { return data_; }

    void resize(uint_t width, uint_t height) {
        data_.resize(width * height);
        width_ = width;
        height_ = height;
    }

    void fill(T value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    Image<T> crop(uint_t x, uint_t y, uint_t w, uint_t h) {
        assert((x+w < width_) && (y+h < height_));
        
        std::vector<T> new_data(w*h);

        for (int nx=0; nx<w; nx++) {
            for (int ny=0; ny<h; ny++) {
                new_data[ny * w + nx] = data_[(y+ny) * width_ + (x+nx)];
            }
        }
        return Image<T>(w,h,new_data);
    }
    
    Image<T> shift(int shift_x, int shift_y, T fill) {
        
        std::vector<T> new_data(width_*height_);

        for (int x=0; x<width_; x++) {
            for (int y=0; y<height_; y++) {
                const int px = x-shift_x;
                const int py = y-shift_y;
                new_data[y * width_ + x] = (0<=px && px<width_) && (0<=py && py<height_) ? data_[ py * width_ + px] : fill;
            }
        }
        return Image<T>(width_, height_, new_data);
    }

    void subsample(float_t scale) {
        assert(scale==0.5);

        uint_t w = width_*0.5;
        uint_t h = height_*0.5;

        std::vector<T> new_data(w*h);

        for (int x=0; x<w; x++) {
            for (int y=0; y<h; y++) {
                new_data[y * w + x] =
                        (data_[(2*y  ) * width_ + (2*x  )] +
                         data_[(2*y+1) * width_ + (2*x  )] + 
                         data_[(2*y  ) * width_ + (2*x+1)] + 
                         data_[(2*y+1) * width_ + (2*x+1)]) * 0.25;
            }
        }
        data_ = std::move(new_data);
        width_ = w;
        height_ = h;
    }


    T at(uint_t x, uint_t y) const {
        assert(x < width_ && y < height_);
        return data_[y * width_ + x];
    }

    Image<float_t> convolution(uint_t kernel_size, std::vector<float_t> kernel ) {
        assert( (kernel_size%2==1) && kernel.size()==(kernel_size*kernel_size));
        
        Image<float_t> result(width_, height_);
        int s = (kernel_size-1)/2;
        
        float_t sum;
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
        assert(width_==img.width_ && height_==img.height_);

        Image<T> result(width_, height_);
        
        for (int w=0; w<width_; w++) {
            for (int h=0; h<height_; h++) {
                result.data()[h * width_ + w] = f( this->data_[h * width_ + w], img.data_[h * width_ + w]);
            }
        }
        return result;    
    }
   
    Image<intensity_t> toIntensity(T min, T max) {
        std::vector<intensity_t> data(width_*height_,0);
        for (int w=0; w<width_; w++) {
            for (int h=0; h<height_; h++) {
                data[h*width_+w] = static_cast<intensity_t>((data_[h * width_ + w]-min)/(max-min)*255.0);
            }
        }
        return Image<intensity_t>(width_, height_, std::begin(data), std::end(data));
    }

    Image<intensity_t> toIntensity() {
        auto mm = std::minmax_element(std::begin(data_), std::end(data_));
        return toIntensity(*(mm.first), *(mm.second));
    }

    cv::Mat exportMat() const {
        cv::Mat img(cv::Size(this->width_, this->height_),CV_8UC1);
        for (int w=0; w<img.cols; w++) {
            for (int h=0; h<img.rows; h++) {
                img.at<unsigned char>(h, w, 0) = data_[h * width_ + w];
            }
        }
        return img;
    }
    
private:

    uint_t width_;
    uint_t height_;
    std::vector<T> data_;

};

#endif /* IMAGE_HPP */
