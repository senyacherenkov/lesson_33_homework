#include "utility.h"
#include <iostream>

std::vector<double> parse_data (std::string& data)
{
    size_t pos = 0;
    size_t newStart = 0;
    std::string token;
    std::vector<double> result;
    while((pos = data.find(SEPARATOR, newStart)) != std::string::npos) {
        token = std::string(std::next(data.begin(), static_cast<long>(newStart)), std::next(data.begin(), static_cast<long>(pos)));
        double value = 0;
        if(!token.empty())
            value = std::stod(token);
        result.push_back(value);
        newStart = pos + 1;
    }
    token = std::string(std::next(data.begin(), static_cast<long>(newStart)), data.end());
    double value = 0;
    if(!token.empty())
        value = std::stod(token);
    result.push_back(value);

    int currentFloor = static_cast<int>(result[result.size() - 2]);
    int wholeBuilding = static_cast<int>(result.back());
    result.erase(std::next(result.begin(), static_cast<long>(result.size() - 1)));
    if(currentFloor == wholeBuilding || currentFloor == FIRST_FLOOR)
        result.back() = 0;
    else
        result.back() = 1;
    return result;
}

void read_data (std::vector<sample_type>& samples)
{
    sample_type m;

    std::string data;
    while(std::getline(std::cin, data))
    {
        std::vector<double> parsedData = parse_data(data);
        assert(parsedData.size() == DATA_LENGTH);

        long i = 0;
        for(const auto & element: parsedData) {
            m(i) = element;
            i++;
        }

        // add this sample to our set of training samples
        samples.push_back(m);
    }
}
