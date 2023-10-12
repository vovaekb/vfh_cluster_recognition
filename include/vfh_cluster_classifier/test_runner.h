#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H

#include <string>

class TestRunner
{
public:
    TestRunner(const std::string& tests_base_path, std::string test_setup_name);

    // Iterate through all the combinations of the values of the test parameters
    void initTests();

    // Run an experiment with the fixed values of test parameters
    // Iterate over all the test scenes and run detector on every single scene
//    void runTestCase(const int test_num);

    void iterateTestScenes();

    // Run the detector on given scene and calculate the evaluation metrics,
    // write the results to the output file
    void runDetector();

private:
    std::string tests_base_path_;
    std::string test_setup_name_;
    std::string test_case_dir_;
    std::string test_output_file_;
    int start_thresh_;
    int end_thresh_;
    std::ofstream output_;
};

#endif // TEST_RUNNER_H
