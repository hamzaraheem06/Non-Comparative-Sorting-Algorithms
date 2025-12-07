echo "Non-Comparison Sorting Algorithms - Chart Generation"


# Check if Python and required packages are available
echo "Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 to generate charts."
    exit 1
fi

# Check for required Python packages
python3 -c "import matplotlib, pandas, seaborn, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required Python packages..."
    pip3 install matplotlib pandas seaborn numpy
fi

# Compile and run the C++ program
echo "🔨 Compiling C++ program..."
g++ -std=c++17 -O2 -o main main.cpp

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"

# Run the program
echo "Running sorting algorithms..."
./main > output.txt

if [ $? -ne 0 ]; then
    echo "Program execution failed!"
    exit 1
fi

echo "Program executed successfully!"

# Generate charts
echo "Generating performance charts..."
python3 generate_charts.py

echo ""
echo "All charts generated successfully!"
echo "Check the 'charts/' directory for generated visualizations"
