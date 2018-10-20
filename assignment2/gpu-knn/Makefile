all: gpu-knn

gpu-knn: ./src/gpu-knn.cu
	/usr/local/cuda/bin/nvcc -o gpu-knn ./src/gpu-knn.cu -Ilibarff ./src/libarff/arff_attr.cpp ./src/libarff/arff_data.cpp ./src/libarff/arff_instance.cpp ./src/libarff/arff_lexer.cpp ./src/libarff/arff_parser.cpp ./src/libarff/arff_scanner.cpp ./src/libarff/arff_token.cpp ./src/libarff/arff_utils.cpp ./src/libarff/arff_value.cpp
