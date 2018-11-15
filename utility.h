#pragma once
#include <string>
#include <vector>
#include <dlib/svm_threaded.h>

using namespace std;
using namespace dlib;

constexpr const char SEPARATOR = ';';
constexpr size_t DATA_LENGTH = 7;
constexpr size_t FIRST_FLOOR = 1;

// Our data will be 7-dimensional data. So declare an appropriate type to contain these points.
typedef dlib::matrix<double,DATA_LENGTH,1> sample_type;
typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;
typedef polynomial_kernel<sample_type> poly_kernel;
typedef radial_basis_kernel<sample_type> rbf_kernel;

class Utility{
private:
    explicit Utility() = default;
    static Utility* instance;
public:
    std::vector<double> parse_data (std::string& data, bool isHandled = false);
    void read_data (std::vector<sample_type>& samples);
public:
    static Utility* getInstance();
};


