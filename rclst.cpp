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

#include "utility.h"
#include <iostream>
#include <vector>
#include <cassert>

#include <dlib/rand.h>

using namespace std;
using namespace dlib;

typedef radial_basis_kernel<sample_type> kernel_type;

// ----------------------------------------------------------------------------------------

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
        std::string filename = argv[2];
        std::vector<sample_type> samples;
        std::vector<sample_type> initial_centers;
        std::vector<double> labels;

        // First, get our labeled set of training data
        read_data(samples);
        cout << "samples.size(): "<< samples.size() << endl;

        //task of classification
        kcentroid<kernel_type> kc(kernel_type(0.1),0.01, 8);
        kkmeans<kernel_type> test(kc);
        test.set_number_of_centers(clusterQuantity);
        pick_initial_centers(static_cast<long>(clusterQuantity), initial_centers, samples, test.get_kernel());
        test.train(samples,initial_centers);

        for (unsigned long i = 0; i < samples.size(); ++i)
            labels.push_back(test(samples[i]));

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
        //cout << "cross validation: \n" << cross_validate_multiclass_trainer(trainer, samples, labels, 1) << endl;
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
        // If you want to save a one_vs_one_decision_function to disk, you can do
        // so.  However, you must declare what kind of decision functions it contains.
        one_vs_one_decision_function<ovo_trainer,
        decision_function<poly_kernel>,  // This is the output of the poly_trainer
        decision_function<rbf_kernel>    // This is the output of the rbf_trainer
        > df2;


        // Put df into df2 and then save df2 to disk.  Note that we could have also said
        // df2 = trainer.train(samples, labels);  But doing it this way avoids retraining.
        df2 = df;
        serialize(filename.c_str()) << df2;
    }
    catch (std::exception& e)
    {
        cout << "exception thrown!" << endl;
        cout << e.what() << endl;
    }
}
