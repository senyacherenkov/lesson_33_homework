#include "utility.h"

#include <iostream>
#include <vector>
#include <cassert>

#include <dlib/rand.h>

typedef dlib::matrix<double,7,1> sample_type;
typedef radial_basis_kernel<sample_type> kernel_type;

int main(int argc, char* argv[])
{
    try
    {
        if (argc != 2)
        {
          std::cerr << "first arg is <filename> of saved data\n";
          return 1;
        }

        one_vs_one_decision_function<ovo_trainer,
        decision_function<poly_kernel>,
        decision_function<rbf_kernel>
        > df;

        // load the function back in from disk and store it in df3.
        deserialize(argv[1]) >> df;

        sample_type m;
        std::string input;
        while(std::getline(std::cin, input)) {
            std::vector<double> parsedData = Utility::getInstance()->parse_data(input, true);
            long i = 0;
            for(const auto & element: parsedData) {
                m(i) = element;
                i++;
            }
        }
    }
    catch (std::exception& e)
    {
        cout << "exception thrown!" << endl;
        cout << e.what() << endl;
    }
}
