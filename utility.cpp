#include "utility.h"
#include <iostream>
#include <fstream>

Utility* Utility::instance = nullptr;
std::vector<double> Utility::parse_data (std::string& data, bool isHandled)
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

    if(!isHandled) {
        int currentFloor = static_cast<int>(result[result.size() - 2]);
        int wholeBuilding = static_cast<int>(result.back());
        result.erase(std::next(result.begin(), static_cast<long>(result.size() - 1)));
        if(currentFloor == wholeBuilding || currentFloor == FIRST_FLOOR)
            result.back() = 0;
        else
            result.back() = 1;
        return result;
    }
    return result;
}

void Utility::read_data (std::vector<sample_type>& samples)
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
            if(i == 2)
                break;
        }

        // add this sample to our set of training samples
        samples.push_back(m);

        if(samples.size() == DATA_LIMIT)
            break;
    }
}

void Utility::create_cluster_files(kkmeans<kernel_type>& algorithm, std::vector<sample_type>& samples, size_t clusterQuantity)
{
    for(size_t i = 0; i < clusterQuantity; i++)
    {
        std::ofstream file(std::to_string(i) + "_data.txt");
        for(const auto& data: samples)
        {
            if(algorithm(data) == i)
                file << data;
        }
        file.close();
    }
}

bool Utility::read_cluster_data(double label)
{
    std::ifstream file;
    std::string filename = std::to_string(label) + "data.txt";
    file.open(filename);
    if(!file.is_open())
        return false;

    std::vector<std::vector<std::string>> fileData;

    std::vector<std::string> tempVector;
    std::string temp;
    size_t count = 0;
    while(std::getline(file, temp)){
        tempVector.push_back(temp);
        if(count == DATA_LENGTH - 1)
        {
            fileData.push_back(tempVector);
            tempVector.clear();
            count = 0;
        }
        count++;

    }
}

Utility *Utility::getInstance()
{
    if (instance == nullptr)
    {
        instance = new Utility();
    }

    return instance;
}
