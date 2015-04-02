#ifndef GNUPLOT_H
#define GNUPLOT_H

#include <iostream>

using namespace std;

class gnuplot {

private:
    FILE *gnuplot_pipe;
    bool is_to_file;

public:
    gnuplot() {
        gnuplot_pipe = popen ("gnuplot -persist", "w");
        is_to_file = false; 
        cout<<"Set up gnuplot.\n";
    }

    gnuplot(const char* filename) {
        gnuplot_pipe = fopen (filename, "w");
        is_to_file = true;
        cout<<"Writing to `"<<filename<<"'\n";  
    }

    ~gnuplot() {
        if ( !is_to_file ) {
            fprintf(gnuplot_pipe, "quit\n");
            pclose(gnuplot_pipe);
        } else {
            fclose(gnuplot_pipe);
        }
    }

    void init_plot( const char* str ){   
        if ( !is_to_file ) {
            fprintf(gnuplot_pipe, "plot '-' using 1:2 with lines \n");
            fprintf(gnuplot_pipe, "set title \"%s\"\n", str);
        }
    }

    void plot_point( double x, double y , double z) {
        fprintf(gnuplot_pipe, "%lf \t%lf \t%lf\n", x, y, z);
        fflush(gnuplot_pipe);
    }

    void plot_point( double x, double y ) {
        fprintf(gnuplot_pipe, "%lf \t%lf\n", x, y);
        fflush(gnuplot_pipe);

    }

    void plot_point( double x ) {
        fprintf(gnuplot_pipe, "%lf\n", x);
        fflush(gnuplot_pipe);

    }
    
    void command( const char* str ) {
        fprintf(gnuplot_pipe, "%s\n", str);
        fflush(gnuplot_pipe);
    }
    
    void finish_plot(  ) {
        if ( !is_to_file ) {
            fprintf(gnuplot_pipe, "e \n");
        }
    }
};
#endif
