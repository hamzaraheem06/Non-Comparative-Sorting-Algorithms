# Non-Comparison Sorting Algorithms Analysis

## Overview

This project implements and analyzes 6 non-comparison sorting algorithms with comprehensive performance testing and visualization:

- **Counting Sort** (Non-Stable & Stable)
- **Radix Sort** (LSD)
- **Bucket Sort**
- **Flash Sort**
- **Spread Sort**

The implementation includes corrected algorithms, performance analysis, automated chart generation.

## Quick Start in VS Code

### Prerequisites

- Visual Studio Code with C/C++ extension
- g++ compiler
- Python 3 (for chart generation)

### Setup

1. **Open Project in VS Code**

   ```bash
   cd /workspace
   code .
   ```

2. **Install C++ Extension**
   - Install **C/C++ Extension Pack** by Microsoft

### Running the Programs

#### Method 1: Using VS Code Terminal

```bash
# Open integrated terminal (Ctrl+` or Cmd+`)
# Compile the program
g++ -std=c++17 -O2 -o main main.cpp

# Run the program
./main
```

#### Method 2: Using Code Runner Extension

- Press `Ctrl+Alt+N` (Windows/Linux) or `Cmd+Alt+N` (macOS)
- Or right-click in the editor and select "Run Code"

#### Method 3: Generate Charts Only

```bash
# Install Python dependencies (if needed)
pip3 install matplotlib pandas seaborn numpy

# Generate performance charts
python3 generate_charts.py
```

### Debugging

1. Set breakpoints by clicking next to line numbers
2. Press `F5` to start debugging

## Files

- **main.cpp** - Main implementation with all 6 algorithms
- **generate_charts.py** - Chart generation script
- **performance_results.csv** - Generated performance data
- **charts/** - Performance visualization charts
- **Report.docx** - Word document report

## Key Features

- **Corrected Implementation** - Fixed CSV generation and memory calculations
- **Comprehensive Testing** - Edge cases, distributions, scalability analysis
- **Automated Charts** - 8 professional visualizations
- **Performance Matrix** - 180 test cases across all algorithms

## Algorithm Selection Guide

| Scenario                | Recommended Algorithm      |
| ----------------------- | -------------------------- |
| Small range (k/n < 0.1) | Counting Sort              |
| Uniform distribution    | Bucket Sort                |
| Fixed-length integers   | Radix Sort                 |
| Unknown characteristics | Spread Sort                |
| In-place sorting needed | Flash Sort                 |
| Stability required      | Stable Counting/Radix Sort |

## Happy Coding.❤️
