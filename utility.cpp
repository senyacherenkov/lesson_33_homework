#include "utility.h"
#include <iostream>
#include <fstream>
#include <map>

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
                for(int i = 0; i < data.size(); i++)
                    file << data(i) << std::endl;
        }
        file.close();
    }
}

bool Utility::read_cluster_data(double label, std::vector<std::vector<std::string>>& fileData)
{
    std::ifstream file;
    std::string filename = std::to_string(static_cast<int>(label)) + "_data.txt";
    file.open(filename);
    if(!file.is_open())
        return false;

    std::vector<std::string> tempVector;
    std::string temp;
    size_t count = 0;
    while(std::getline(file, temp)){
        tempVector.push_back(temp);
        if(count == DATA_LENGTH - 1)
        {
            fileData.push_back(tempVector);
            tempVector.clear();            
        }
        count++;
        if(count == DATA_LENGTH)
            count = 0;
    }
    return true;
}

void Utility::sort_and_display_cluster_data(double x, double y, std::vector<std::vector<string> > &fileData)
{
    std::map<double, std::vector<std::string>> result;
    for(const auto& apartment: fileData) {
        double diffX = x - std::stod(apartment.at(0));
        double diffY = y - std::stod(apartment.at(1));
        result.emplace(sqrt(pow(diffX, 2) + pow(diffY, 2)), apartment);
    }

    for(const auto& resultPair: result) {
        std::string output;
        for(const auto & str: resultPair.second)
        {
            output += str; output += ';';
        }
        output.back() = '\n';
        std::cout << output;
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
