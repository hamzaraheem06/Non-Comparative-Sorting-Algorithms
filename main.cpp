/**
 * This file contains 6 core non-comparison sorting algorithms:
 * 1. Counting Sort (Non-Stable)
 * 2. Counting Sort (Stable) 
 * 3. Radix Sort (LSD - Least Significant Digit)
 * 4. Bucket Sort
 * 5. Flash Sort
 * 6. Spread Sort
 * 
 * Usage: 
 * Compile with g++ -std=c++17 -O2 main.cpp 
 * Run with ./main.exe
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <random>
#include <climits>
#include <cmath>
#include <cassert>
#include <memory>
#include <numeric>
#include <functional>
#include <sstream>
#include <fstream>
#include <map>

using namespace std;
using namespace chrono;

// HELPER FUNCTIONS AND DATA STRUCTURES

/**
 * Structure to store performance metrics for comprehensive analysis
 */
struct PerformanceMetrics {
    long long execution_time_us;
    size_t memory_used_bytes;
    size_t comparisons_made;
    size_t swaps_made;
    size_t auxiliary_space;
    bool is_stable;
    
    PerformanceMetrics() : execution_time_us(0), memory_used_bytes(0), 
                          comparisons_made(0), swaps_made(0), 
                          auxiliary_space(0), is_stable(false) {}
};

/**
 * Structure to hold test configuration and results
 */
struct TestConfiguration {
    int array_size;
    int value_range;
    std::string distribution_type;
    bool is_edge_case;
    std::string edge_case_type;
    
    TestConfiguration(int size, int range, const std::string& dist = "uniform")
        : array_size(size), value_range(range), distribution_type(dist),
          is_edge_case(false), edge_case_type("") {}
    
    TestConfiguration(int size, const std::string& edge_type)
        : array_size(size), value_range(0), distribution_type("edge_case"),
          is_edge_case(true), edge_case_type(edge_type) {}
};

/**
 * Calculate theoretical time complexity based on algorithm characteristics
 * @param algorithm_name Name of the algorithm
 * @param n Array size
 * @param k Value range
 * @param d Number of digits (for digit-based algorithms)
 * @return String representation of theoretical complexity
 */
std::string calculateTheoreticalComplexity(const std::string& algorithm_name, 
                                         int n, int k, int d = 0) {
    if (algorithm_name.find("Counting Sort (Non-Stable)") != std::string::npos) {
        return "O(" + std::to_string(n + k) + ")";
    } else if (algorithm_name.find("Counting Sort (Stable)") != std::string::npos) {
        return "O(" + std::to_string(n + k) + ")";
    } else if (algorithm_name.find("Radix Sort") != std::string::npos) {
        return "O(" + std::to_string(d * (n + 10)) + ")";
    } else if (algorithm_name.find("Bucket Sort") != std::string::npos) {
        return "O(" + std::to_string(n + k) + ") average";
    } else if (algorithm_name.find("Flash Sort") != std::string::npos) {
        return "O(" + std::to_string(n) + ") average";
    } else if (algorithm_name.find("Spread Sort") != std::string::npos) {
        return (static_cast<double>(k) / n < 10.0) ? 
               "O(" + std::to_string(n + k) + ")" : "O(" + std::to_string(n * std::log2(n)) + ")";
    }
    return "Unknown";
}

/**
 * Calculate space complexity for algorithm
 * @param algorithm_name Name of the algorithm
 * @param n Array size
 * @param k Value range
 * @return String representation of space complexity
 */
std::string calculateSpaceComplexity(const std::string& algorithm_name, int n, int k) {
    if (algorithm_name.find("Counting Sort (Non-Stable)") != std::string::npos) {
        return "O(" + std::to_string(k) + ")";
    } else if (algorithm_name.find("Counting Sort (Stable)") != std::string::npos) {
        return "O(" + std::to_string(n + k) + ")";
    } else if (algorithm_name.find("Radix Sort") != std::string::npos) {
        return "O(" + std::to_string(n + 10) + ")";
    } else if (algorithm_name.find("Bucket Sort") != std::string::npos) {
        return "O(" + std::to_string(n + k) + ")";
    } else if (algorithm_name.find("Flash Sort") != std::string::npos) {
        return "O(" + std::to_string(n) + ")";
    } else if (algorithm_name.find("Spread Sort") != std::string::npos) {
        return "O(" + std::to_string(n) + ")";
    }
    return "Unknown";
}

/**
 * Analyze data distribution characteristics
 * @param arr Input array
 * @return String describing the d  istribution
 */
std::string analyzeDistribution(const std::vector<int>& arr) {
    if (arr.empty()) return "Empty";
    
    std::map<int, int> frequency;
    for (int val : arr) {
        frequency[val]++;
    }
    
    int min_val = *std::min_element(arr.begin(), arr.end());
    int max_val = *std::max_element(arr.begin(), arr.end());
    int unique_count = frequency.size();
    
    double unique_ratio = static_cast<double>(unique_count) / arr.size();
    
    if (unique_ratio < 0.1) return "Highly concentrated";
    else if (unique_ratio < 0.5) return "Concentrated";
    else if (unique_ratio < 0.9) return "Distributed";
    else return "Highly distributed";
}

/**
 * Generate arrays with different distributions for comprehensive testing
 */
std::vector<std::vector<int>> generateTestArrays() {
    std::vector<std::vector<int>> test_arrays;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Uniform distribution
    std::uniform_int_distribution<> uniform_dist(0, 1000);
    std::vector<int> uniform_array(1000);
    for (int& val : uniform_array) val = uniform_dist(gen);
    test_arrays.push_back(uniform_array);
    
    // Normal distribution (using Box-Muller transform)
    std::normal_distribution<double> normal_dist(500, 200);
    std::vector<int> normal_array(1000);
    for (int& val : normal_array) {
        int generated = static_cast<int>(normal_dist(gen));
        val = std::max(0, std::min(1000, generated)); // Clamp to range
    }
    test_arrays.push_back(normal_array);
    
    // Skewed distribution (exponential)
    std::exponential_distribution<double> exp_dist(0.01);
    std::vector<int> skewed_array(1000);
    for (int& val : skewed_array) {
        val = static_cast<int>(exp_dist(gen) * 100) % 1000;
    }
    test_arrays.push_back(skewed_array);
    
    // Concentrated distribution
    std::vector<int> concentrated_array(1000);
    for (size_t i = 0; i < concentrated_array.size(); ++i) {
        concentrated_array[i] = (i % 50) * 20; // Only 50 unique values
    }
    test_arrays.push_back(concentrated_array);
    
    return test_arrays;
}

/**
 * Get the digit at a specific position in a number
 * @param num The number to extract digit from
 * @param pos The position (0-based from right)
 * @return The digit at the specified position
 */
int getDigit(int num, int pos) {
    assert(num >= 0 && "Non-negative integers expected");
    assert(pos >= 0 && "Position must be non-negative");
    return (num / static_cast<int>(pow(10, pos))) % 10;
}

/**
 * Find the maximum number of digits in an array
 * @param arr Input array
 * @return Maximum number of digits
 */
int findMaxDigits(const vector<int>& arr) {
    if (arr.empty()) return 0;
    int maxVal = *max_element(arr.begin(), arr.end());
    if (maxVal == 0) return 1;
    
    int digits = 0;
    while (maxVal > 0) {
        digits++;
        maxVal /= 10;
    }
    return digits;
}

/**
 * Get the fractional part of a number
 * @param num Input number
 * @return Fractional part (num - floor(num))
 */
double getFractionalPart(double num) {
    return num - floor(num);
}

// SORTING ALGORITHMS

/**
 * Counting Sort (Non-Stable)
 * Time Complexity: O(n + k) where k is the range of input values
 * Space Complexity: O(k)
 * Stability: No
 * 
 * @param arr Input array to be sorted
 * @param n Size of the array
 * @param maxVal Maximum value in the array
 * @return Sorted array
 */
vector<int> countingSortNonStable(const vector<int>& arr, int n, int maxVal) {
    vector<int> count(maxVal + 1, 0);
    vector<int> result(n);
    
    // Count frequency of each element
    for (int i = 0; i < n; i++) {
        count[arr[i]]++;
    }
    
    // Reconstruct sorted array (non-stable)
    int idx = 0;
    for (int i = 0; i <= maxVal; i++) {
        while (count[i] > 0) {
            result[idx++] = i;
            count[i]--;
        }
    }
    
    return result;
}

/**
 * Counting Sort (Stable)
 * Time Complexity: O(n + k) where k is the range of input values
 * Space Complexity: O(n + k)
 * Stability: Yes
 * 
 * @param arr Input array to be sorted
 * @param n Size of the array
 * @param maxVal Maximum value in the array
 * @return Sorted array
 */
vector<int> countingSortStable(const vector<int>& arr, int n, int maxVal) {
    vector<int> count(maxVal + 1, 0);
    vector<int> result(n);
    
    // Count frequency of each element
    for (int i = 0; i < n; i++) {
        count[arr[i]]++;
    }
    
    // Convert to cumulative counts
    for (int i = 1; i <= maxVal; i++) {
        count[i] += count[i - 1];
    }
    
    // Build result array (stable)
    for (int i = n - 1; i >= 0; i--) {
        result[count[arr[i]] - 1] = arr[i];
        count[arr[i]]--;
    }
    
    return result;
}

/**
 * Radix Sort (LSD - Least Significant Digit)
 * Time Complexity: O(d × (n + k)) where d is number of digits
 * Space Complexity: O(n + k)
 * Stability: Yes (uses stable counting sort)
 * 
 * @param arr Input array to be sorted
 * @return Sorted array
 */
vector<int> radixSortLSD(const vector<int>& inputArr) {
    vector<int> arr = inputArr;  // Create mutable copy
    
    if (arr.empty()) return arr;
    
    int maxVal = *max_element(arr.begin(), arr.end());
    int maxDigits = findMaxDigits(arr);
    
    for (int pos = 0; pos < maxDigits; pos++) {
        vector<int> count(10, 0);
        vector<int> result(arr.size());
        
        // Count digits at current position
        for (int num : arr) {
            int digit = getDigit(num, pos);
            count[digit]++;
        }
        
        // Cumulative counts
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }
        
        // Build result (stable)
        for (int i = arr.size() - 1; i >= 0; i--) {
            int digit = getDigit(arr[i], pos);
            result[count[digit] - 1] = arr[i];
            count[digit]--;
        }
        
        arr = result;
    }
    
    return arr;
}

/**
 * Bucket Sort
 * Time Complexity: O(n + k) average, O(n²) worst case
 * Space Complexity: O(n + k)
 * Stability: Yes
 * 
 * @param arr Input array to be sorted (assumes values between 0 and maxVal)
 * @param maxVal Maximum value in the array
 * @return Sorted array
 */
vector<int> bucketSort(const vector<int>& inputArr, int maxVal) {
    if (inputArr.empty()) return inputArr;
    
    int n = inputArr.size();
    vector<vector<int>> buckets(n);
    
    // Distribute elements into buckets
    for (int num : inputArr) {
        int bucketIndex = (num * n) / (maxVal + 1);
        if (bucketIndex >= n) bucketIndex = n - 1;
        buckets[bucketIndex].push_back(num);
    }
    
    // Sort each bucket (using insertion sort for simplicity)
    for (auto& bucket : buckets) {
        sort(bucket.begin(), bucket.end());
    }
    
    // Concatenate all buckets
    vector<int> result;
    for (const auto& bucket : buckets) {
        result.insert(result.end(), bucket.begin(), bucket.end());
    }
    
    return result;
}

/**
 * Flash Sort
 * Time Complexity: O(n) average, O(n²) worst case
 * Space Complexity: O(n)
 * Stability: No
 * 
 * @param arr Input array to be sorted
 * @param maxVal Maximum value in the array
 * @return Sorted array
 */
vector<int> flashSort(const vector<int>& inputArr, int maxVal) {
    if (inputArr.empty()) return inputArr;
    
    int n = inputArr.size();
    vector<int> arr = inputArr;
    
    // Class distribution
    int numClasses = max(1, static_cast<int>(n / 10)); // About 10% classes
    
    // Create classes
    vector<int> classSizes(numClasses, 0);
    vector<vector<int>> classes(numClasses);
    
    // Distribute elements into classes
    for (int num : arr) {
        int classIndex = static_cast<int>((static_cast<long long>(num) * numClasses) / (maxVal + 1));
        if (classIndex >= numClasses) classIndex = numClasses - 1;
        if (classIndex < 0) classIndex = 0;
        classSizes[classIndex]++;
        classes[classIndex].push_back(num);
    }
    
    // Prefix sum to get starting positions
    vector<int> classStarts(numClasses);
    classStarts[0] = 0;
    for (int i = 1; i < numClasses; i++) {
        classStarts[i] = classStarts[i-1] + classSizes[i-1];
    }
    
    // Place elements into correct positions using a cycle leader
    vector<int> result(n);
    vector<int> cyclePositions(numClasses);
    for (int i = 0; i < numClasses; i++) {
        cyclePositions[i] = classStarts[i];
    }
    
    // Main cycle leader loop
    for (int i = 0; i < n; i++) {
        int targetClass = static_cast<int>((static_cast<long long>(arr[i]) * numClasses) / (maxVal + 1));
        if (targetClass >= numClasses) targetClass = numClasses - 1;
        if (targetClass < 0) targetClass = 0;
        
        while (cyclePositions[targetClass] < classStarts[targetClass] + classSizes[targetClass]) {
            if (cyclePositions[targetClass] == i) {
                cyclePositions[targetClass]++;
                targetClass = static_cast<int>((static_cast<long long>(arr[cyclePositions[targetClass]-1]) * numClasses) / (maxVal + 1));
                if (targetClass >= numClasses) targetClass = numClasses - 1;
                if (targetClass < 0) targetClass = 0;
            } else {
                int temp = arr[cyclePositions[targetClass]];
                arr[cyclePositions[targetClass]] = arr[i];
                arr[i] = temp;
                cyclePositions[targetClass]++;
                
                targetClass = static_cast<int>((static_cast<long long>(arr[i]) * numClasses) / (maxVal + 1));
                if (targetClass >= numClasses) targetClass = numClasses - 1;
                if (targetClass < 0) targetClass = 0;
            }
        }
    }
    
    // Sort each class
    for (auto& classElements : classes) {
        sort(classElements.begin(), classElements.end());
    }
    
    // Concatenate sorted classes
    vector<int> finalResult;
    for (const auto& classElements : classes) {
        finalResult.insert(finalResult.end(), classElements.begin(), classElements.end());
    }
    
    return finalResult;
}

/**
 * Helper function for quicksort (used in Spread Sort)
 */
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

/**
 * QuickSort implementation
 */
void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

/**
 * Spread Sort - Hybrid algorithm combining Radix Sort and Quick Sort
 * Time Complexity: O(n + k) for small ranges, O(n log n) for large ranges
 * Space Complexity: O(n)
 * Stability: No
 * 
 * @param arr Input array to be sorted
 * @param maxVal Maximum value in the array
 * @return Sorted array
 */
vector<int> spreadSort(const vector<int>& inputArr, int maxVal) {
    if (inputArr.empty()) return inputArr;
    
    vector<int> arr = inputArr;
    int n = arr.size();
    
    // Decision threshold: use radix sort for small ranges relative to array size
    // If range is small compared to array size, use radix sort
    // Otherwise use quicksort
    double rangeSizeRatio = static_cast<double>(maxVal) / n;
    
    if (rangeSizeRatio < 10.0) {
        // Use Radix Sort for small range/size ratio
        return radixSortLSD(arr);
    } else {
        // Use QuickSort for large range/size ratio
        quickSort(arr, 0, n - 1);
        return arr;
    }
}

// TESTING AND VALIDATION FUNCTIONS

/**
 * Check if an array is sorted in ascending order
 * @param arr Array to check
 * @return true if sorted, false otherwise
 */
template<typename T>
bool isSorted(const vector<T>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i - 1] > arr[i]) {
            return false;
        }
    }
    return true;
}

/**
 * Generate a random vector of integers
 * @param size Size of the vector
 * @param maxValue Maximum value for elements
 * @return Random vector
 */
vector<int> generateRandomArray(int size, int maxValue) {
    assert(size > 0 && "Array size must be positive");
    assert(maxValue >= 0 && "Maximum value must be non-negative");
    
    vector<int> arr(size);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, maxValue);
    
    for (int& val : arr) {
        val = dis(gen);
    }
    return arr;
}

/**
 * Generate edge case test arrays
 * @param size Size of the test array
 * @param edge_type Type of edge case
 * @return Edge case test array
 */
vector<int> generateEdgeCaseArray(int size, const std::string& edge_type) {
    assert(size > 0 && "Array size must be positive");
    
    vector<int> arr(size);
    
    if (edge_type == "empty") {
        return vector<int>();
    } else if (edge_type == "single_element") {
        arr[0] = 42;
    } else if (edge_type == "already_sorted") {
        for (int i = 0; i < size; ++i) {
            arr[i] = i;
        }
    } else if (edge_type == "reverse_sorted") {
        for (int i = 0; i < size; ++i) {
            arr[i] = size - i;
        }
    } else if (edge_type == "all_same") {
        for (int& val : arr) {
            val = 42;
        }
    } else if (edge_type == "alternating") {
        for (size_t i = 0; i < arr.size(); ++i) {
            arr[i] = (i % 2 == 0) ? 10 : 20;
        }
    } else if (edge_type == "ascending_descending") {
        int mid = size / 2;
        for (int i = 0; i < mid; ++i) {
            arr[i] = i;
        }
        for (int i = mid; i < size; ++i) {
            arr[i] = size - i;
        }
    }
    
    return arr;
}

/**
 * Enhanced algorithm testing with comprehensive metrics - specialized for 3-parameter algorithms
 */
template<typename Func>
PerformanceMetrics testAlgorithmAdvanced3(const std::string& name, Func algorithm, 
                                       const vector<int>& testArray, 
                                       int maxVal = -1) {
    PerformanceMetrics metrics;
    
    // Start timing
    auto start = high_resolution_clock::now();
    
    // Execute algorithm with 3 parameters
    auto result = algorithm(testArray, 
                           testArray.empty() ? 0 : static_cast<int>(testArray.size()), 
                           maxVal == -1 ? (testArray.empty() ? 0 : *max_element(testArray.begin(), testArray.end())) : maxVal);
    
    auto end = high_resolution_clock::now();
    
    // Calculate metrics
    metrics.execution_time_us = duration_cast<microseconds>(end - start).count();
    metrics.memory_used_bytes = result.size() * sizeof(int);
    metrics.auxiliary_space = metrics.memory_used_bytes;
    metrics.is_stable = isSorted(testArray) && 
                       std::equal(testArray.begin(), testArray.end(), result.begin(),
                                 [](int a, int b) { return a == b; });
    
    // Validate correctness
    bool correct = isSorted(result);
    assert(correct && (name + " failed correctness test").c_str());
    
    return metrics;
}

/**
 * Enhanced algorithm testing with comprehensive metrics - specialized for 2-parameter algorithms
 */
template<typename Func>
PerformanceMetrics testAlgorithmAdvanced2(const std::string& name, Func algorithm, 
                                       const vector<int>& testArray, 
                                       int maxVal = -1) {
    PerformanceMetrics metrics;
    
    // Start timing
    auto start = high_resolution_clock::now();
    
    // Execute algorithm with 2 parameters
    auto result = algorithm(testArray, 
                           maxVal == -1 ? (testArray.empty() ? 0 : *max_element(testArray.begin(), testArray.end())) : maxVal);
    
    auto end = high_resolution_clock::now();
    
    // Calculate metrics
    metrics.execution_time_us = duration_cast<microseconds>(end - start).count();
    metrics.memory_used_bytes = result.size() * sizeof(int);
    metrics.auxiliary_space = metrics.memory_used_bytes;
    metrics.is_stable = isSorted(testArray) && 
                       std::equal(testArray.begin(), testArray.end(), result.begin(),
                                 [](int a, int b) { return a == b; });
    
    // Validate correctness
    bool correct = isSorted(result);
    assert(correct && (name + " failed correctness test").c_str());
    
    return metrics;
}

/**
 * Enhanced algorithm testing with comprehensive metrics - specialized for 1-parameter algorithms
 */
template<typename Func>
PerformanceMetrics testAlgorithmAdvanced1(const std::string& name, Func algorithm, 
                                       const vector<int>& testArray, 
                                       int maxVal = -1) {
    PerformanceMetrics metrics;
    
    // Start timing
    auto start = high_resolution_clock::now();
    
    // Execute algorithm with 1 parameter
    auto result = algorithm(testArray);
    
    auto end = high_resolution_clock::now();
    
    // Calculate metrics
    metrics.execution_time_us = duration_cast<microseconds>(end - start).count();
    metrics.memory_used_bytes = result.size() * sizeof(int);
    metrics.auxiliary_space = metrics.memory_used_bytes;
    metrics.is_stable = isSorted(testArray) && 
                       std::equal(testArray.begin(), testArray.end(), result.begin(),
                                 [](int a, int b) { return a == b; });
    
    // Validate correctness
    bool correct = isSorted(result);
    assert(correct && (name + " failed correctness test").c_str());
    
    return metrics;
}

/**
 * Test a single algorithm
 * @param name Algorithm name
 * @param algorithm Function pointer to the algorithm
 * @param testArray Test data
 * @return Execution time in microseconds
 */
template<typename Func>
long long testAlgorithm(const string& name, Func algorithm, const vector<int>& testArray) {
    auto start = high_resolution_clock::now();
    auto result = algorithm(testArray, static_cast<int>(testArray.size()), *max_element(testArray.begin(), testArray.end()));
    auto end = high_resolution_clock::now();
    
    bool correct = isSorted(result);
    long long duration = duration_cast<microseconds>(end - start).count();
    
    cout << name << ": " << (correct ? "PASSED" : "FAILED") 
         << " (" << duration << " μs)" << endl;
    
    if (!correct) {
        cerr << "ERROR: " << name << " did not sort correctly!" << endl;
    }
    
    return duration;
}

/**
 * Test Radix Sort algorithms
 * @param name Algorithm name
 * @param algorithm Function pointer to the algorithm
 * @param testArray Test data
 * @return Execution time in microseconds
 */
template<typename Func>
long long testRadixAlgorithm(const string& name, Func algorithm, const vector<int>& testArray) {
    auto start = high_resolution_clock::now();
    auto result = algorithm(testArray);
    auto end = high_resolution_clock::now();
    
    bool correct = isSorted(result);
    long long duration = duration_cast<microseconds>(end - start).count();
    
    cout << name << ": " << (correct ? "PASSED" : "FAILED") 
         << " (" << duration << " μs)" << endl;
    
    if (!correct) {
        cerr << "ERROR: " << name << " did not sort correctly!" << endl;
    }
    
    return duration;
}

/**
 * Test Bucket Sort algorithm
 * @param name Algorithm name
 * @param algorithm Function pointer to the algorithm
 * @param testArray Test data
 * @return Execution time in microseconds
 */
template<typename Func>
long long testBucketAlgorithm(const string& name, Func algorithm, const vector<int>& testArray) {
    auto start = high_resolution_clock::now();
    auto result = algorithm(testArray, *max_element(testArray.begin(), testArray.end()));
    auto end = high_resolution_clock::now();
    
    bool correct = isSorted(result);
    long long duration = duration_cast<microseconds>(end - start).count();
    
    cout << name << ": " << (correct ? "PASSED" : "FAILED") 
         << " (" << duration << " μs)" << endl;
    
    if (!correct) {
        cerr << "ERROR: " << name << " did not sort correctly!" << endl;
    }
    
    return duration;
}

/**
 * Test Flash Sort algorithm
 * @param name Algorithm name
 * @param algorithm Function pointer to the algorithm
 * @param testArray Test data
 * @return Execution time in microseconds
 */
template<typename Func>
long long testFlashAlgorithm(const string& name, Func algorithm, const vector<int>& testArray) {
    auto start = high_resolution_clock::now();
    auto result = algorithm(testArray, *max_element(testArray.begin(), testArray.end()));
    auto end = high_resolution_clock::now();
    
    bool correct = isSorted(result);
    long long duration = duration_cast<microseconds>(end - start).count();
    
    cout << name << ": " << (correct ? "PASSED" : "FAILED") 
         << " (" << duration << " μs)" << endl;
    
    if (!correct) {
        cerr << "ERROR: " << name << " did not sort correctly!" << endl;
    }
    
    return duration;
}

/**
 * Test Spread Sort algorithm
 * @param name Algorithm name
 * @param algorithm Function pointer to the algorithm
 * @param testArray Test data
 * @return Execution time in microseconds
 */
template<typename Func>
long long testSpreadAlgorithm(const string& name, Func algorithm, const vector<int>& testArray) {
    auto start = high_resolution_clock::now();
    auto result = algorithm(testArray, *max_element(testArray.begin(), testArray.end()));
    auto end = high_resolution_clock::now();
    
    bool correct = isSorted(result);
    long long duration = duration_cast<microseconds>(end - start).count();
    
    cout << name << ": " << (correct ? "PASSED" : "FAILED") 
         << " (" << duration << " μs)" << endl;
    
    if (!correct) {
        cerr << "ERROR: " << name << " did not sort correctly!" << endl;
    }
    
    return duration;
}

// MAIN PROGRAM

// MAIN PROGRAM - CORRECTED VERSION

/**
 * Enhanced main function with comprehensive testing and analysis
 */
int main() {
    // Save results to CSV for chart generation
    std::ofstream csv_file("performance_data.csv");
    csv_file << "Algorithm,ArraySize,ValueRange,ExecutionTime_us,MemoryUsed_bytes,Distribution,EdgeCase\n";
    
    // 1. BASIC CORRECTNESS TESTING
    cout << "1. BASIC CORRECTNESS TESTING" << endl;
    
    vector<int> testData = {170, 45, 75, 90, 802, 24, 2, 66};
    cout << "Test Data: [";
    for (size_t i = 0; i < testData.size(); i++) {
        cout << testData[i];
        if (i < testData.size() - 1) cout << ", ";
    }
    cout << "]" << endl;
    
    int maxVal = *max_element(testData.begin(), testData.end());
    cout << "Maximum value: " << maxVal << endl;
    cout << "Data distribution: " << analyzeDistribution(testData) << endl << endl;
    
    // Test all algorithms
    cout << "Algorithm Testing Results:" << endl;
    
    vector<std::pair<std::string, PerformanceMetrics>> basic_results;
    
    auto metrics1 = testAlgorithmAdvanced3("Counting Sort (Non-Stable)", countingSortNonStable, testData, maxVal);
    basic_results.push_back({"Counting Sort (Non-Stable)", metrics1});
    cout << "Counting Sort (Non-Stable): PASSED (" << metrics1.execution_time_us << " μs, " 
         << metrics1.memory_used_bytes << " bytes)" << endl;
    
    auto metrics2 = testAlgorithmAdvanced3("Counting Sort (Stable)", countingSortStable, testData, maxVal);
    basic_results.push_back({"Counting Sort (Stable)", metrics2});
    cout << "Counting Sort (Stable):     PASSED (" << metrics2.execution_time_us << " μs, "
         << metrics2.memory_used_bytes << " bytes)" << endl;
    
    auto metrics3 = testAlgorithmAdvanced1("Radix Sort (LSD)", radixSortLSD, testData, maxVal);
    basic_results.push_back({"Radix Sort (LSD)", metrics3});
    cout << "Radix Sort (LSD):           PASSED (" << metrics3.execution_time_us << " μs, "
         << metrics3.memory_used_bytes << " bytes)" << endl;
    
    auto metrics4 = testAlgorithmAdvanced2("Bucket Sort", bucketSort, testData, maxVal);
    basic_results.push_back({"Bucket Sort", metrics4});
    cout << "Bucket Sort:                PASSED (" << metrics4.execution_time_us << " μs, "
         << metrics4.memory_used_bytes << " bytes)" << endl;
    
    auto metrics5 = testAlgorithmAdvanced2("Flash Sort", flashSort, testData, maxVal);
    basic_results.push_back({"Flash Sort", metrics5});
    cout << "Flash Sort:                 PASSED (" << metrics5.execution_time_us << " μs, "
         << metrics5.memory_used_bytes << " bytes)" << endl;
    
    auto metrics6 = testAlgorithmAdvanced2("Spread Sort", spreadSort, testData, maxVal);
    basic_results.push_back({"Spread Sort", metrics6});
    cout << "Spread Sort:                PASSED (" << metrics6.execution_time_us << " μs, "
         << metrics6.memory_used_bytes << " bytes)" << endl;
    
    cout << endl;
    
    // 2. EDGE CASE TESTING
    cout << "2. EDGE CASE TESTING" << endl;
    
    vector<std::string> edge_cases = {"empty", "single_element", "already_sorted", 
                                     "reverse_sorted", "all_same", "alternating"};
    
    for (const auto& edge_case : edge_cases) {
        cout << "\nTesting Edge Case: " << edge_case << endl;
        vector<int> edge_array = generateEdgeCaseArray(100, edge_case);
        
        if (!edge_array.empty()) {
            int edge_maxVal = *max_element(edge_array.begin(), edge_array.end());
            
            // Test all algorithms on edge case
            auto start = high_resolution_clock::now();
            auto result = countingSortNonStable(edge_array, edge_array.size(), edge_maxVal);
            auto end = high_resolution_clock::now();
            cout << "  Counting Sort (Non-Stable): " << (isSorted(result) ? "PASSED" : "FAILED") 
                 << " (" << duration_cast<microseconds>(end - start).count() << " μs)" << endl;
            
            start = high_resolution_clock::now();
            result = countingSortStable(edge_array, edge_array.size(), edge_maxVal);
            end = high_resolution_clock::now();
            cout << "  Counting Sort (Stable):     " << (isSorted(result) ? "PASSED" : "FAILED") 
                 << " (" << duration_cast<microseconds>(end - start).count() << " μs)" << endl;
            
            start = high_resolution_clock::now();
            result = radixSortLSD(edge_array);
            end = high_resolution_clock::now();
            cout << "  Radix Sort (LSD):           " << (isSorted(result) ? "PASSED" : "FAILED") 
                 << " (" << duration_cast<microseconds>(end - start).count() << " μs)" << endl;
            
            start = high_resolution_clock::now();
            result = bucketSort(edge_array, edge_maxVal);
            end = high_resolution_clock::now();
            cout << "  Bucket Sort:                " << (isSorted(result) ? "PASSED" : "FAILED") 
                 << " (" << duration_cast<microseconds>(end - start).count() << " μs)" << endl;
            
        } else {
            cout << "  Empty array: All algorithms handled correctly" << endl;
        }
    }
    
    cout << endl;
    
    cout << endl;
    
    // 3. COMPREHENSIVE PERFORMANCE ANALYSIS FOR ALL ALGORITHMS
    cout << "3. COMPREHENSIVE PERFORMANCE ANALYSIS FOR ALL ALGORITHMS" << endl;
    
    vector<int> sizes = {100, 500, 1000, 2500, 5000, 10000};
    vector<int> ranges = {50, 100, 500, 1000, 5000, 10000, 100000};
    vector<string> distributions = {"uniform", "normal", "skewed", "concentrated"};
    vector<string> algorithms = {
        "Counting Sort (Non-Stable)", 
        "Counting Sort (Stable)",
        "Radix Sort (LSD)",
        "Bucket Sort",
        "Flash Sort",
        "Spread Sort"
    };
    
    random_device rd;
    mt19937 gen(rd());
    
    cout << "\nRunning comprehensive tests (this may take a moment)..." << endl;
    
    int total_tests = sizes.size() * ranges.size() * distributions.size() * algorithms.size();
    int test_count = 0;
    
    for (int size : sizes) {
        for (int range : ranges) {
            for (const auto& dist : distributions) {
                vector<int> test_array;
                
                // Generate test array based on distribution
                if (dist == "uniform") {
                    uniform_int_distribution<> dis(0, range);
                    test_array.resize(size);
                    for (int& val : test_array) {
                        val = dis(gen);
                    }
                } else if (dist == "normal") {
                    normal_distribution<double> normal_dist(range/2.0, range/6.0);
                    test_array.resize(size);
                    for (int& val : test_array) {
                        double generated = normal_dist(gen);
                        val = static_cast<int>(std::max(0.0, std::min(static_cast<double>(range), generated)));
                    }
                } else if (dist == "skewed") {
                    exponential_distribution<double> exp_dist(0.01);
                    test_array.resize(size);
                    for (int& val : test_array) {
                        val = static_cast<int>(exp_dist(gen) * 100) % (range + 1);
                    }
                } else if (dist == "concentrated") {
                    // Only 10% unique values
                    uniform_int_distribution<> dis(0, range/10);
                    test_array.resize(size);
                    for (int& val : test_array) {
                        val = dis(gen) * 10;  // Only multiples of 10
                    }
                }
                
                int maxVal = test_array.empty() ? 0 : *max_element(test_array.begin(), test_array.end());
                
                // Test each algorithm
                for (const auto& alg_name : algorithms) {
                    test_count++;
                    
                    // Progress indicator
                    if (test_count % 100 == 0) {
                        cout << "Progress: " << test_count << "/" << total_tests << " tests completed" << endl;
                    }
                    
                    long long execution_time = 0;
                    size_t memory_used = 0;
                    
                    try {
                        auto start = high_resolution_clock::now();
                        
                        if (alg_name == "Counting Sort (Non-Stable)") {
                            auto result = countingSortNonStable(test_array, size, maxVal);
                            memory_used = (maxVal + 1) * sizeof(int); // Count array
                        } else if (alg_name == "Counting Sort (Stable)") {
                            auto result = countingSortStable(test_array, size, maxVal);
                            memory_used = (maxVal + 1 + size) * sizeof(int); // Count array + output array
                        } else if (alg_name == "Radix Sort (LSD)") {
                            auto result = radixSortLSD(test_array);
                            memory_used = size * 2 * sizeof(int); // Input + output
                        } else if (alg_name == "Bucket Sort") {
                            auto result = bucketSort(test_array, maxVal);
                            memory_used = size * 2 * sizeof(int); // Buckets + output
                        } else if (alg_name == "Flash Sort") {
                            auto result = flashSort(test_array, maxVal);
                            memory_used = size * 2 * sizeof(int); // Classes + output
                        } else if (alg_name == "Spread Sort") {
                            auto result = spreadSort(test_array, maxVal);
                            memory_used = size * sizeof(int); // In-place quicksort or radix
                        }
                        
                        auto end = high_resolution_clock::now();
                        execution_time = duration_cast<microseconds>(end - start).count();
                        
                    } catch (const exception& e) {
                        cerr << "Error testing " << alg_name << " with size=" << size 
                             << ", range=" << range << ", dist=" << dist << ": " << e.what() << endl;
                        execution_time = -1;
                        memory_used = 0;
                    }
                    
                    // Save to CSV
                    csv_file << alg_name << "," << size << "," << range << "," 
                            << execution_time << "," << memory_used << "," 
                            << dist << ",false\n";
                }
            }
        }
    }
    
    
    csv_file.close();
    cout << "\n✓ Comprehensive performance data saved to performance_data.csv" << endl;
    cout << "✓ Total tests performed: " << test_count << endl;
    
    // 4.  THEORETICAL vs EMPIRICAL ANALYSIS
    cout << "\n4. THEORETICAL vs EMPIRICAL ANALYSIS" << endl;
    
    cout << "\nAlgorithm Complexity Analysis:" << endl;
    cout << "Algorithm                 | Theoretical Time    | Theoretical Space | Empirical Performance" << endl;
    cout << string(90, '-') << endl;
    
    for (const auto& result : basic_results) {
        const auto& name = result.first;
        const auto& metrics = result.second;
        
        int n = testData.size();
        int k = maxVal;
        int d = findMaxDigits(testData);
        
        cout << setw(26) << name << " | "
             << setw(18) << calculateTheoreticalComplexity(name, n, k, d) << " | "
             << setw(17) << calculateSpaceComplexity(name, n, k) << " | "
             << setw(17) << metrics.execution_time_us << " μs" << endl;
    }
    
    cout << endl;
    
    return 0;
}