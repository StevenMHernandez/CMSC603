#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;

#define K_VALUE 5

__device__ float euclideanDistance(float *x, float *xi, int num_attributes) {
    float sum = 0;

    /*
     * for each attribute (not counting the class)
     */
    for (int a = 0; a < num_attributes; a++) {
        sum += pow(x[a] - xi[a], 2);
    }

    return sqrt(sum);
}

//d_input, d_output, d_num_attributes, d_k_value
__global__ void knn(float *input, int *output, int *num_attributes, int *k_value, int *num_instances, int *max_class_number) {
	int tid = (blockDim.x * blockIdx.x) + threadIdx.x;

	int num_indices_per_item = *num_attributes + 1;

	if (tid < *num_instances) {
        /*
         * Create arrays to store the k nearest neighbors (storing their class and distance
         */
        int *neighbors_class = (int *) malloc(*k_value * sizeof(int));
        float *neighbors_distance = (float *) malloc(*k_value * sizeof(float));

        for (int n = 0; n < *k_value; n++) { // todo: this could have been a `memset`
            neighbors_class[n] = 0;
            neighbors_distance[n] = FLT_MAX;
        }


        /*
         * compare to all other data in the dataset
         */
        for (int j = 0; j < *num_instances; j++) {
            // do not compare to itself
            if (tid != j) {
                int classValue = input[((j + 1) * num_indices_per_item) - 1];

				float *x = &(input[tid * num_indices_per_item]);
				float *xi = &(input[j * num_indices_per_item]);

                float distance = euclideanDistance(x, xi, *num_attributes);

                // sort into neighbors array
                for (int n = 0; n < *k_value; n++) {
                    if (distance < neighbors_distance[n]) {
                        int tempClassValue = neighbors_class[n];
                        float tempClassDistance = neighbors_distance[n];

                        neighbors_class[n] = classValue;
                        neighbors_distance[n] = distance;

                        classValue = tempClassValue;
                        distance = tempClassDistance;
                    }
                }
            }
        }

        // count votes
        int *votes = (int *) malloc((*max_class_number + 1) * sizeof(int));

        // for each voted value
		for (int n = 0; n < *max_class_number + 1; n++) {
			votes[n] = 0;
		}
        for (int n = 0; n < *k_value; n++) {
            votes[neighbors_class[n]] += 1;
        }

        int max_class = 0;

        for (int v = 0; v < *max_class_number + 1; v++) {
            if (votes[max_class] < votes[v]) {
                max_class = v;
            }
        }

        free(votes);
        free(neighbors_class);
        free(neighbors_distance);

        output[tid] = max_class;
    }
}

int* KNN(ArffData* dataset, int k_value)
{
	/*
     * Find the maximum number of classes in the data.
     * This is used to determine the highest voted class per data point.
     */
	int max_class_number = 0;

	for (int i = 0; i < dataset->num_instances(); i++) {
		int classValue = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();

		if (classValue > max_class_number) {
			max_class_number = classValue;
		}
	}

	/**
	 * CUDA Specific code
	 */

	int number_of_attributes = dataset->get_instance(0)->size() - 1;

	float *h_input, *d_input;
	int *h_output, *d_output, *d_k_value, *d_num_attributes, *d_num_instances, *d_max_class_number;

	cudaMalloc(&d_num_attributes, sizeof(int));
	cudaMalloc(&d_k_value, sizeof(int));
    cudaMalloc(&d_num_instances, sizeof(int));
    cudaMalloc(&d_max_class_number, sizeof(int));
	cudaMalloc(&d_input, (number_of_attributes + 1) * dataset->num_instances() * sizeof(float));
	cudaMalloc(&d_output, dataset->num_instances() * sizeof(float));

	h_input = (float *) malloc(number_of_attributes * dataset->num_instances() * sizeof(float));
	h_output = (int *) malloc(dataset->num_instances() * sizeof(int));

	for (int i = 0; i < dataset->num_instances(); i++) {
		ArffInstance *x = dataset->get_instance(i);

		for (int a = 0; a < x->size(); a++) {
			h_input[(i * x->size()) + a] = x->get(a)->operator float();
		}
	}

	int num_instances = dataset->num_instances();

	cudaMemcpy(d_num_attributes, &number_of_attributes, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_value, &k_value, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_num_instances, &num_instances, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_max_class_number, &max_class_number, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_input, h_input, (number_of_attributes + 1) * dataset->num_instances() * sizeof(float), cudaMemcpyHostToDevice);


	/// Run kernel
	int threadsPerBlock = 1;
	int blocksPerGrid = ceil((double) dataset->num_instances() / (double) threadsPerBlock);
	printf("Running kernel with %i blocks (%i threads each) for %i instances\n", blocksPerGrid, threadsPerBlock, dataset->num_instances());
	knn<<<blocksPerGrid,threadsPerBlock>>>(d_input, d_output, d_num_attributes, d_k_value, d_num_instances, d_max_class_number);


	cudaMemcpy(h_output, d_output, dataset->num_instances() * sizeof(int), cudaMemcpyDeviceToHost);

	return h_output;
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
	int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses

	for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
	{
		int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
		int predictedClass = predictions[i];

		confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
	}

	return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
	int successfulPredictions = 0;

	for(int i = 0; i < dataset->num_classes(); i++)
	{
		successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
	}

	return successfulPredictions / (float) dataset->num_instances();
}

int main(int argc, char *argv[])
{
	// TODO: re-add
//	if(argc != 2)
//	{
//		cout << "Usage: ./main datasets/datasetFile.arff" << endl;
//		exit(0);
//	}
//
//	ArffParser parser(argv[1]);
//	ArffParser parser("../datasets/medium.arff");
	ArffParser parser("../datasets/small.arff");
	ArffData *dataset = parser.parse();
	struct timespec start, end;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);

	int* predictions = KNN(dataset, K_VALUE);
	int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
	float accuracy = computeAccuracy(confusionMatrix, dataset);

	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

	printf("The KNN classifier for %lu instances required %llu ms CPU time. Accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}
