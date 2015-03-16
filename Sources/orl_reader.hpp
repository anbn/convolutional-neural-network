#ifndef ORL_READER_HPP
#define ORL_READER_HPP

#include "utils.hpp"
#include "image.hpp"

namespace nn {

class orl_reader {
    
public:

    uint_t num_examples() const { return num_examples_; }

    const Image<float_t>& image(uint_t n) { return images_[n]; }
    int   label(uint_t n) { return labels_[n]; }

    orl_reader() : num_examples_(0) {}

    bool read(std::string path, uint_t num) {
        std::cout<<"Reading orl... ";
        
        if (num == 0 || num > 400) {
            std::cout<<"\nSetting number of images to maximum of 400.\n";
            num = 400;
        }

        images_.resize(num);
        labels_.resize(num);
        
        std::ifstream infile;
        std::string s;
        int width, height;
        unsigned char byte;

        for (int f=0; f<10 && num_examples_<num; f++) {
            for (int n=0; n<40 && num_examples_<num; n++) {
                infile.open(path+"/s"+std::to_string(n+1)+"/"+std::to_string(f+1)+".pgm",
                        std::ios::binary | std::ios::in);
                if(!infile.is_open())
                    throw "Error opening file.";

                std::getline(infile, s);
                if (s.compare("P5"))
                    throw "P5 Format required";

                std::getline(infile, s);
                width = std::stoi(s.substr(0, s.find(" ")));
                height = std::stoi(s.substr(s.find(" ")));

                std::getline(infile, s);
                std::vector<float_t> data(width*height);

                for (int h=0; h<height; h++) {
                    for (int w=0; w<width; w++) {
                        infile.read(reinterpret_cast<char*>(&byte), sizeof(unsigned char));
                        data[h*width+w] = byte/255.0; /* scale to [0,1] */
                    }
                }
                infile.close();

                int idx = f*40+n;
                images_[idx] = Image<float_t>(width, height, data);
                images_[idx].crop(6,30,80,80);
                labels_[idx] = 1;

                ++num_examples_;
            }
        }

        std::cout<<"("<<num_examples_<<" images read)\n";
        return true;
    }

protected:

    uint_t num_examples_;
        
    std::vector< Image<float_t> > images_;
    std::vector<int> labels_;
};

void orl_reader_test() {
    orl_reader orl;
    orl.read("data/orl_faces", 100);

    for (int i=0; i<orl.num_examples(); i++) {
        Image<nn::float_t> img = orl.image(i);
        cv::imshow("orl["+std::to_string(i)+"]", img.toIntensity(0,1).exportMat());
        while(cv::waitKey(0)!=27);
    }
}



} /* namespace nn */

#endif /* ORL_READER.HPP */

