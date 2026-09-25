# CUDA Clutter Generator

### Visual Studio Setup

1. Create a new, empty C++ project
2. Title the project "*radarClutterCuda*"
3. Create the solution folder inside the root directory
4. Right-click the project, select Build Dependencies -> Build Customizations...
5. Ensure CUDA is ticked
6. Right-click the project, select Add -> Existing Item...
7. Add *cudaClutterMode.cu*
8. Add each *.cu* file inside the *kernel/* directory
9. Right-click the project, select Properties
10. Under Configuration Properties -> CUDA C/C++ -> Common, Add "*$(SolutionDir)..\\kernel\\include;%(AdditionalIncludeDirectories)*" to Additional Include Directories





## To-do



1. Implement a random number generator using CUDA
2. Take timing measurements for CPU \& GPU implementations
3. Create probability distribution tests

