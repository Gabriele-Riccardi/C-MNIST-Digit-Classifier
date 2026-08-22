#ifndef MAIN_H
#define MAIN_H
/*
MNIST section.
MNIST file formats:
Images: The first 16 bytes are the header, the next bytes are pixel values.
    Offset 0: magic number (should be 2051)
    Offset 4: number of images
    Offset 8: number of rows
    Offset 12: number of columns
    Offset 16: image pixel data begins
Labels: The first 8 bytes are the header, the next bytes are the labels.
    Offset 0: magic number (should be 2049)
    Offset 4: number of items
    Offset 8: label data begins
*/
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <time.h>

#define MAX_HIDDEN_NODES 512
#define MAX_OUTPUT_NODES 16

typedef struct {
    int size;
    int *labels;
    float **images;
} mnist_data;

int reverse_int(int i);
mnist_data* load_mnist_images(const char* images_filename);
mnist_data* load_mnist_labels(const char* labels_filename);

// Neural Network architecture section.
typedef struct {
    int input_nodes;
    int output_nodes;
    int hidden_nodes;
    float *w1, *b1;
    float *w2, *b2;
    float *hidden;
} neural_net;

neural_net* initialize_network(int input_nodes, int output_nodes, int hidden_nodes);

// Propagation section.
void forward_propagation(neural_net *net, float *input, float *output);
void softmax(float *input, int size);
void forward_propagate_with_activation(neural_net *net, float *input, float *output);

void backpropagation(neural_net *net, float *input, int *target, float *output, float lr);
void shift_image(const float *src, float *dst, int rows, int cols, int dx, int dy);
void train(neural_net *net, mnist_data *train_data, mnist_data *train_labels, int epochs, float learning_rate);
float cross_entropy_loss(float *output, int *target, int size);

// Action section.
float evaluate(neural_net *net, mnist_data *test_data, mnist_data *test_labels);

// Save and load seaction.
int save_network(neural_net *net, const char *filename);
int load_network(neural_net *net, const char *filename);

//Free functions section,
void free_network(neural_net *net);

int reverse_int(int i) {
    unsigned char c1 =  i        & 255;
    unsigned char c2 = (i >> 8)  & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

#endif