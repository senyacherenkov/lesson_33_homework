#pragma once
#include <string>
#include <vector>
#include <dlib/svm_threaded.h>

constexpr const char SEPARATOR = ';';
constexpr size_t DATA_LENGTH = 7;
constexpr size_t FIRST_FLOOR = 1;

// Our data will be 2-dimensional data. So declare an appropriate type to contain these points.
typedef dlib::matrix<double,DATA_LENGTH,1> sample_type;


std::vector<double> parse_data (std::string& data);
void read_data (std::vector<sample_type>& samples);
