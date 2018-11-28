#pragma once
#include <string>
#include <vector>
#include <dlib/svm_threaded.h>

using namespace std;
using namespace dlib;

constexpr const char SEPARATOR = ';';
constexpr size_t DATA_LENGTH = 7;
constexpr size_t FIRST_FLOOR = 1;
constexpr size_t DATA_LIMIT = 1000;

// Our data will be 7-dimensional data. So declare an appropriate type to contain these points.
typedef dlib::matrix<double,7,1> sample_type;
typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;
typedef polynomial_kernel<sample_type> poly_kernel;
typedef radial_basis_kernel<sample_type> rbf_kernel;
typedef radial_basis_kernel<sample_type> kernel_type;

class Utility{
private:
    explicit Utility() = default;
    static Utility* instance;
public:
    std::vector<double> parse_data (std::string& data, bool isHandled = false);
    void read_data (std::vector<sample_type>& samples);
    void create_cluster_files(kkmeans<kernel_type>& algorithm, std::vector<sample_type>& samples, size_t clusterQuantity);
    bool read_cluster_data(double label, std::vector<std::vector<string> > &fileData);
    void sort_and_display_cluster_data(double x, double y, std::vector<std::vector<string> > &fileData);
public:
    static Utility* getInstance();
};


