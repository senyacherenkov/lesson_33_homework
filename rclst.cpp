// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the multiclass classification tools
    from the dlib C++ Library.  Specifically, this example will make points from
    three classes and show you how to train a multiclass classifier to recognize
    these three classes.

    The classes are as follows:
        - class 1: points very close to the origin
        - class 2: points on the circle of radius 10 around the origin
        - class 3: points that are on a circle of radius 4 but not around the origin at all
*/

#include <dlib/svm_threaded.h>

#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <dlib/rand.h>

using namespace std;
using namespace dlib;

constexpr const char SEPARATOR = ';';
constexpr size_t DATA_LENGTH = 7;

// Our data will be 2-dimensional data. So declare an appropriate type to contain these points.
typedef matrix<double,DATA_LENGTH,1> sample_type;
typedef radial_basis_kernel<sample_type> kernel_type;

// ----------------------------------------------------------------------------------------

void read_data (std::vector<sample_type>& samples);

std::vector<double> parse_data (std::string& data)
{
    size_t pos = 0;
    size_t newStart = 0;
    std::string token;
    std::vector<double> result;
    while((pos = data.find(SEPARATOR, newStart)) != std::string::npos) {
        token = std::string(std::next(data.begin(), static_cast<long>(newStart)), std::next(data.begin(), static_cast<long>(pos)));
        result.push_back(std::stoi(token));
        newStart = pos + 1;
    }
    int currentFloor = static_cast<int>(result[result.size() - 2]);
    int wholeBuilding = static_cast<int>(result.back());
    result.erase(std::next(result.begin(), static_cast<long>(result.size() - 1)));
    if(currentFloor == wholeBuilding || currentFloor == 1)
        result.back() = 0;
    else
        result.back() = 1;
    return result;
}
// ----------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    try
    {
        if (argc != 3)
        {
          std::cerr << "first arg is <amount> of clusters, second is <filename> of saved data\n";
          return 1;
        }

        size_t clusterQuantity = static_cast<size_t>(std::stoi(argv[1]));
        std::vector<sample_type> samples;
        std::vector<sample_type> initial_centers;

        // First, get our labeled set of training data
        read_data(samples);
        cout << "samples.size(): "<< samples.size() << endl;

        //task of classification
        kcentroid<kernel_type> kc(kernel_type(0.1),0.01, 8);
        kkmeans<kernel_type> test(kc);
        test.set_number_of_centers(clusterQuantity);
        pick_initial_centers(static_cast<long>(clusterQuantity), initial_centers, samples, test.get_kernel());
        test.train(samples,initial_centers);
        // The main object in this example program is the one_vs_one_trainer.  It is essentially
        // a container class for regular binary classifier trainer objects.  In particular, it
        // uses the any_trainer object to store any kind of trainer object that implements a
        // .train(samples,labels) function which returns some kind of learned decision function.
        // It uses these binary classifiers to construct a voting multiclass classifier.  If
        // there are N classes then it trains N*(N-1)/2 binary classifiers, one for each pair of
        // labels, which then vote on the label of a sample.
        //
        // In this example program we will work with a one_vs_one_trainer object which stores any
        // kind of trainer that uses our sample_type samples.
        typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;


        // Finally, make the one_vs_one_trainer.
        ovo_trainer trainer;


        // Next, we will make two different binary classification trainer objects.  One
        // which uses kernel ridge regression and RBF kernels and another which uses a
        // support vector machine and polynomial kernels.  The particular details don't matter.
        // The point of this part of the example is that you can use any kind of trainer object
        // with the one_vs_one_trainer.
        typedef polynomial_kernel<sample_type> poly_kernel;
        typedef radial_basis_kernel<sample_type> rbf_kernel;

        // make the binary trainers and set some parameters
        krr_trainer<rbf_kernel> rbf_trainer;
        svm_nu_trainer<poly_kernel> poly_trainer;
        poly_trainer.set_kernel(poly_kernel(0.1, 1, 2));
        rbf_trainer.set_kernel(rbf_kernel(0.1));


        // Now tell the one_vs_one_trainer that, by default, it should use the rbf_trainer
        // to solve the individual binary classification subproblems.
        trainer.set_trainer(rbf_trainer);
        // We can also get more specific.  Here we tell the one_vs_one_trainer to use the
        // poly_trainer to solve the class 1 vs class 2 subproblem.  All the others will
        // still be solved with the rbf_trainer.
        trainer.set_trainer(poly_trainer, 1, 2);

        // Now let's do 5-fold cross-validation using the one_vs_one_trainer we just setup.
        // As an aside, always shuffle the order of the samples before doing cross validation.
        // For a discussion of why this is a good idea see the svm_ex.cpp example.
        randomize_samples(samples, labels);
        cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 5) << endl;
        // The output is shown below.  It is the confusion matrix which describes the results.  Each row
        // corresponds to a class of data and each column to a prediction.  Reading from top to bottom,
        // the rows correspond to the class labels if the labels have been listed in sorted order.  So the
        // top row corresponds to class 1, the middle row to class 2, and the bottom row to class 3.  The
        // columns are organized similarly, with the left most column showing how many samples were predicted
        // as members of class 1.
        //
        // So in the results below we can see that, for the class 1 samples, 60 of them were correctly predicted
        // to be members of class 1 and 0 were incorrectly classified.  Similarly, the other two classes of data
        // are perfectly classified.
        /*
            cross validation:
            60  0  0
            0 70  0
            0  0 80
        */

        // Next, if you wanted to obtain the decision rule learned by a one_vs_one_trainer you
        // would store it into a one_vs_one_decision_function.
        one_vs_one_decision_function<ovo_trainer> df = trainer.train(samples, labels);

        cout << "predicted label: "<< df(samples[0])  << ", true label: "<< labels[0] << endl;
        cout << "predicted label: "<< df(samples[90]) << ", true label: "<< labels[90] << endl;
        // The output is:
        /*
            predicted label: 2, true label: 2
            predicted label: 1, true label: 1
        */


        // If you want to save a one_vs_one_decision_function to disk, you can do
        // so.  However, you must declare what kind of decision functions it contains.
        one_vs_one_decision_function<ovo_trainer,
        decision_function<poly_kernel>,  // This is the output of the poly_trainer
        decision_function<rbf_kernel>    // This is the output of the rbf_trainer
        > df2, df3;


        // Put df into df2 and then save df2 to disk.  Note that we could have also said
        // df2 = trainer.train(samples, labels);  But doing it this way avoids retraining.
        df2 = df;
        serialize("df.dat") << df2;

        // load the function back in from disk and store it in df3.
        deserialize("df.dat") >> df3;


        // Test df3 to see that this worked.
        cout << endl;
        cout << "predicted label: "<< df3(samples[0])  << ", true label: "<< labels[0] << endl;
        cout << "predicted label: "<< df3(samples[90]) << ", true label: "<< labels[90] << endl;
        // Test df3 on the samples and labels and print the confusion matrix.
        cout << "test deserialized function: \n" << test_multiclass_decision_function(df3, samples, labels) << endl;





        // Finally, if you want to get the binary classifiers from inside a multiclass decision
        // function you can do it by calling get_binary_decision_functions() like so:
        one_vs_one_decision_function<ovo_trainer>::binary_function_table functs;
        functs = df.get_binary_decision_functions();
        cout << "number of binary decision functions in df: " << functs.size() << endl;
        // The functs object is a std::map which maps pairs of labels to binary decision
        // functions.  So we can access the individual decision functions like so:
        decision_function<poly_kernel> df_1_2 = any_cast<decision_function<poly_kernel> >(functs[make_unordered_pair(1,2)]);
        decision_function<rbf_kernel>  df_1_3 = any_cast<decision_function<rbf_kernel>  >(functs[make_unordered_pair(1,3)]);
        // df_1_2 contains the binary decision function that votes for class 1 vs. 2.
        // Similarly, df_1_3 contains the classifier that votes for 1 vs. 3.

        // Note that the multiclass decision function doesn't know what kind of binary
        // decision functions it contains.  So we have to use any_cast to explicitly cast
        // them back into the concrete type.  If you make a mistake and try to any_cast a
        // binary decision function into the wrong type of function any_cast will throw a
        // bad_any_cast exception.
    }
    catch (std::exception& e)
    {
        cout << "exception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

void generate_data (
    std::vector<sample_type>& samples,
    std::vector<double>& labels
)
{
    sample_type m;

    dlib::rand rnd;

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
        labels.push_back(1);
    }
}

// ----------------------------------------------------------------------------------------
