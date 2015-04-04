#ifndef ORL_READER_HPP
#define ORL_READER_HPP

#include "utils.hpp"
#include "image.hpp"

namespace nn {

class orl_reader {
    
public:

    uint_t num_examples() const { return images_.size(); }

    const Image<float_t>& image(uint_t n) { return images_[n]; }
    int   label(uint_t n) { return labels_[n]; }

    orl_reader() {}

    bool read(std::string path, uint_t num) {
        std::cout<<"Reading orl... ";
        
        if (num == 0 || num > 400) {
            std::cout<<"\nSetting number of images to maximum of 400.\n";
            num = 400;
        }

        std::ifstream infile;
        std::string s;
        int width, height;
        unsigned char byte;

        for (int f=0; f<10 && num>0; f++) {
            for (int n=0; n<40 && num>0; n++) {
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
                images_.push_back(Image<float_t>(width, height, data));
                images_.back().crop(14,38,64,64);
                images_.back().subsample(0.5);
                labels_.push_back(1);

                --num;
            }
        }


        std::cout<<"("<<num_examples()<<" images read)\n";
        return true;
    }

    bool generate_counterexamples(uint_t num, uint_t dim, std::string filename) {
        Image<float_t> img (cv::imread(filename));
        //cv::imshow("test",img.exportMat());
        std::cout<<"Generating counterexmaples... ";

        for (int n=0; n<num; n++) {
            int x = get_random(0.0, (double)(img.width()-dim));
            int y = get_random(0.0, (double)(img.height()-dim));

            std::vector<float_t> data(dim*dim);
            for (int h=0; h<dim; h++) {
                for (int w=0; w<dim; w++) {
                    data[h*dim+w] = img.at(x+w,y+h)/255.0;
                }
            }
            images_.push_back(Image<float_t>(dim, dim, data));
            labels_.push_back(-1);
        }
        std::cout<<"("<<num_examples()<<" images read)\n";
    }


protected:

    std::vector< Image<float_t> > images_;
    std::vector<int> labels_;
};

void orl_reader_test() {
    orl_reader orl;

    orl.read("data/orl_faces", 10);
    orl.generate_counterexamples(10, 100, "data/misc/saopaulo.jpg");
    

    for (int i=0; i<orl.num_examples(); i++) {
        Image<nn::float_t> img = orl.image(i);
        std::cout<<"orl["<<i<<"]: "<<img.width()<<"x"<<img.height()<<"\n";
        cv::imshow("orl["+std::to_string(i)+"]", img.toIntensity(0,1).exportMat());
        while(cv::waitKey(0)!=27);
    }
}



} /* namespace nn */

#endif /* ORL_READER.HPP */

