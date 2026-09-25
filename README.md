# CUDA Clutter Generator

### Visual Studio Setup

1. Create a new, empty C++ project
1. Create the solution folder inside the root directory
1. Right-click the project, select Build Dependencies -> Build Customizations...
1. Ensure CUDA is ticked
1. Right-click the project, select Add -> Existing Item...
1. Add _cudaClutterMode.cu_
1. Add each _.cu_ file inside the _kernel/_ directory
1. Right-click the project, select Properties
1. Under Configuration Properties -> CUDA C/C++ -> Common, Add "_$(SolutionDir)..\kernel\include;%(AdditionalIncludeDirectories)_" to Additional Include Directories