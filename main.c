#include "main.h"
#include "prng.h"

/* ---------------- MNIST image loader ---------------- */

mnist_data* load_mnist_images(const char* images_filename) {
    FILE* fp = fopen(images_filename, "rb");
    if (!fp) {
        perror("fopen");
        return NULL;
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

    /* IDX stores integers big-endian, so every header field is byte-swapped. */
    if (fread(&magic_number, sizeof(magic_number), 1, fp) != 1) {
        fprintf(stderr, "Truncated header in %s\n", images_filename);
        fclose(fp);
        return NULL;
    }
    magic_number = reverse_int(magic_number);

    if (magic_number != 2051) {
        fprintf(stderr, "Invalid MNIST image file: magic %d (expected 2051)\n", magic_number);
        fclose(fp);
        return NULL;
    }

    if (fread(&num_images, sizeof(num_images), 1, fp) != 1 ||
        fread(&num_rows,   sizeof(num_rows),   1, fp) != 1 ||
        fread(&num_cols,   sizeof(num_cols),   1, fp) != 1) {
        fprintf(stderr, "Truncated header in %s\n", images_filename);
        fclose(fp);
        return NULL;
    }
    num_images = reverse_int(num_images);
    num_rows   = reverse_int(num_rows);
    num_cols   = reverse_int(num_cols);

    mnist_data* data = (mnist_data*)malloc(sizeof(mnist_data));
    if (!data) {
        fprintf(stderr, "Failed to allocate mnist_data\n");
        fclose(fp);
        return NULL;
    }

    data->size = num_images;
    data->images = (float**)calloc(num_images, sizeof(float*));
    if (!data->images) {
        fprintf(stderr, "Failed to allocate the pointer array\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    /* One scratch buffer reused for every image: the file stores raw bytes,
       which are converted to normalised floats after the read. */
    const size_t pixels = (size_t)num_rows * num_cols;
    unsigned char *row = (unsigned char*)malloc(pixels);
    if (!row) {
        fprintf(stderr, "Failed to allocate the read buffer\n");
        free(data->images);
        free(data);
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < num_images; i++) {
        data->images[i] = (float*)malloc(pixels * sizeof(float));

        if (!data->images[i]) {
            fprintf(stderr, "Allocation failed at image %d of %d\n", i, num_images);
            for (int q = 0; q < i; q++) free(data->images[q]);
            free(row);
            free(data->images);
            free(data);
            fclose(fp);
            return NULL;
        }

        /* One read per image instead of one per pixel: 784x fewer calls. */
        if (fread(row, 1, pixels, fp) != pixels) {
            fprintf(stderr, "Truncated pixel data at image %d\n", i);
            for (int q = 0; q <= i; q++) free(data->images[q]);
            free(row);
            free(data->images);
            free(data);
            fclose(fp);
            return NULL;
        }

        /* Scale to [0,1]: unnormalised inputs make the gradients blow up. */
        for (size_t p = 0; p < pixels; p++)
            data->images[i][p] = row[p] / 255.0f;
    }

    free(row);
    fclose(fp);
    return data;
}

/* ---------------- MNIST label loader ---------------- */

mnist_data* load_mnist_labels(const char* labels_filename) {
    FILE* fp = fopen(labels_filename, "rb");
    if (!fp) {
        perror("fopen");
        return NULL;
    }

    int magic_number = 0, num_labels = 0;

    if (fread(&magic_number, sizeof(magic_number), 1, fp) != 1) {
        fprintf(stderr, "Truncated header in %s\n", labels_filename);
        fclose(fp);
        return NULL;
    }
    magic_number = reverse_int(magic_number);

    if (magic_number != 2049) {
        fprintf(stderr, "Invalid MNIST label file: magic %d (expected 2049)\n", magic_number);
        fclose(fp);
        return NULL;
    }

    if (fread(&num_labels, sizeof(num_labels), 1, fp) != 1) {
        fprintf(stderr, "Truncated header in %s\n", labels_filename);
        fclose(fp);
        return NULL;
    }
    num_labels = reverse_int(num_labels);

    mnist_data* data = (mnist_data*)malloc(sizeof(mnist_data));
    if (!data) {
        fprintf(stderr, "Failed to allocate mnist_data\n");
        fclose(fp);
        return NULL;
    }

    data->size = num_labels;
    data->labels = (int*)malloc(num_labels * sizeof(int));
    if (!data->labels) {
        fprintf(stderr, "Failed to allocate the label array\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    unsigned char *buf = (unsigned char*)malloc((size_t)num_labels);
    if (!buf || fread(buf, 1, (size_t)num_labels, fp) != (size_t)num_labels) {
        fprintf(stderr, "Truncated label data in %s\n", labels_filename);
        free(buf);
        free(data->labels);
        free(data);
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < num_labels; i++)
        data->labels[i] = (int)buf[i];

    free(buf);
    fclose(fp);
    return data;
}

/* ---------------- network initialisation ---------------- */

neural_net* initialize_network(int input_nodes, int output_nodes, int hidden_nodes) {
    /* backpropagation() keeps its delta buffers on the stack at a fixed size,
       so refuse any geometry that would overrun them. */
    if (hidden_nodes > MAX_HIDDEN_NODES || output_nodes > MAX_OUTPUT_NODES ||
        hidden_nodes <= 0 || output_nodes <= 0 || input_nodes <= 0) {
        fprintf(stderr, "Unsupported network geometry: %d-%d-%d (limits: %d hidden, %d output)\n",
                input_nodes, hidden_nodes, output_nodes,
                MAX_HIDDEN_NODES, MAX_OUTPUT_NODES);
        return NULL;
    }

    neural_net *net = (neural_net*)malloc(sizeof(neural_net));
    if (net == NULL) {
        perror("Memory allocation failed");
        return NULL;
    }

    net->input_nodes  = input_nodes;
    net->output_nodes = output_nodes;
    net->hidden_nodes = hidden_nodes;

    net->w1     = (float*)malloc((size_t)hidden_nodes * input_nodes  * sizeof(float));
    net->b1     = (float*)calloc((size_t)hidden_nodes,                 sizeof(float));
    net->w2     = (float*)malloc((size_t)output_nodes * hidden_nodes * sizeof(float));
    net->b2     = (float*)calloc((size_t)output_nodes,                 sizeof(float));
    net->hidden = (float*)malloc((size_t)hidden_nodes                * sizeof(float));

    if (!net->w1 || !net->b1 || !net->w2 || !net->b2 || !net->hidden) {
        fprintf(stderr, "Failed to allocate the network parameters\n");
        free_network(net);
        return NULL;
    }

    /* First layer: Xavier/Glorot uniform, scaled on the fan-in. */
    float limit1 = sqrtf(6.0f / (float)input_nodes);
    for (size_t i = 0; i < (size_t)hidden_nodes * input_nodes; i++) {
        float r = prng_randf();
        net->w1[i] = (r * 2.0f - 1.0f) * limit1;
    }

    /* Second layer: Xavier as well, since it feeds into softmax. */
    float limit2 = sqrtf(6.0f / (float)(hidden_nodes + output_nodes));
    for (size_t i = 0; i < (size_t)output_nodes * hidden_nodes; i++) {
        float r = prng_randf();
        net->w2[i] = (r * 2.0f - 1.0f) * limit2;
    }

    return net;
}

/* ---------------- forward pass ---------------- */

/* Each hidden unit takes a dot product over the 784 inputs plus its bias and
   applies ReLU; the output layer produces ten raw logits. */
void forward_propagation(neural_net *net, float *input, float *output) {
    for (int h = 0; h < net->hidden_nodes; h++) {
        const float *w = &net->w1[h * net->input_nodes];
        float sum = net->b1[h];
        for (int in = 0; in < net->input_nodes; in++)
            sum += input[in] * w[in];
        net->hidden[h] = sum > 0.0f ? sum : 0.0f;      /* ReLU */
    }

    for (int o = 0; o < net->output_nodes; o++) {
        const float *w = &net->w2[o * net->hidden_nodes];
        float sum = net->b2[o];
        for (int h = 0; h < net->hidden_nodes; h++)
            sum += net->hidden[h] * w[h];
        output[o] = sum;
    }
}

/* Turns the raw logits into a probability distribution. The maximum is
   subtracted before exponentiating: without it, large logits overflow. */
void softmax(float *output, int size) {
    float max = output[0];
    for (int i = 1; i < size; i++) {
        if (output[i] > max) {
            max = output[i];
        }
    }

    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        output[i] = expf(output[i] - max);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void forward_propagate_with_activation(neural_net *net, float *input, float *output) {
    forward_propagation(net, input, output);
    softmax(output, net->output_nodes);
}

/* Only the target class contributes to the loss. The probability is clamped
   away from zero so that a confident mistake does not produce log(0). */
float cross_entropy_loss(float *output, int *target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        if (target[i] == 1) {
            loss -= logf(fmaxf(output[i], 1e-7f));
        }
    }

    return loss;
}

/* ---------------- backward pass ----------------
 *
 * Gradient descent: every weight moves against the gradient of the loss,
 * scaled by the learning rate.
 *
 * Chain rule: the error is propagated backwards layer by layer. For softmax
 * combined with cross-entropy the output-layer gradient collapses to
 * (output - target), which is why delta2 is a single subtraction.
 *
 * Learning rate: controls the step size. Too small and training crawls, too
 * large and the updates overshoot the minimum.
 */
void backpropagation(neural_net *net, float *input, int *target, float *output, float lr) {
    float delta2[MAX_OUTPUT_NODES];
    float delta1[MAX_HIDDEN_NODES];

    for (int o = 0; o < net->output_nodes; o++)
        delta2[o] = output[o] - target[o];

    /* --- delta1 must be computed with the OLD w2 weights --- */
    for (int h = 0; h < net->hidden_nodes; h++)
        delta1[h] = 0.0f;

    for (int o = 0; o < net->output_nodes; o++) {
        const float d = delta2[o];
        const float *w = &net->w2[o * net->hidden_nodes];
        for (int h = 0; h < net->hidden_nodes; h++)
            delta1[h] += d * w[h];
    }

    for (int h = 0; h < net->hidden_nodes; h++)
        if (net->hidden[h] <= 0.0f) delta1[h] = 0.0f;   /* ReLU derivative */

    /* --- only now is it safe to update the weights --- */
    for (int o = 0; o < net->output_nodes; o++) {
        const float lrd = lr * delta2[o];
        float *w = &net->w2[o * net->hidden_nodes];
        for (int h = 0; h < net->hidden_nodes; h++)
            w[h] -= lrd * net->hidden[h];
        net->b2[o] -= lrd;
    }

    for (int h = 0; h < net->hidden_nodes; h++) {
        const float lrd = lr * delta1[h];
        if (lrd == 0.0f) continue;                      /* dead ReLU, nothing to do */
        float *w = &net->w1[h * net->input_nodes];
        for (int in = 0; in < net->input_nodes; in++)
            w[in] -= lrd * input[in];
        net->b1[h] -= lrd;
    }
}

/* Translates the image by (dx, dy), padding the exposed border with zeros. */
void shift_image(const float *src, float *dst, int rows, int cols, int dx, int dy) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int sr = r - dy, sc = c - dx;
            dst[r * cols + c] = (sr >= 0 && sr < rows && sc >= 0 && sc < cols) ? src[sr * cols + sc] : 0.0f;
        }
    }
}

/* ---------------- training loop ---------------- */

void train(neural_net *net, mnist_data *train_data, mnist_data *train_labels, int epochs, float learning_rate) {
    float *output  = (float*)malloc(net->output_nodes * sizeof(float));
    float *shifted = (float*)malloc(28 * 28 * sizeof(float));
    int   *order   = (int*)malloc(train_data->size * sizeof(int));
    if (!output || !shifted || !order) { free(output); free(shifted); free(order); return; }

    for (int i = 0; i < train_data->size; i++) order[i] = i;

    for (int epoch = 0; epoch < epochs; epoch++) {
        clock_t t0 = clock();
        float total_loss = 0.0f;

        /* Fisher-Yates shuffle: a fixed presentation order biases the updates. */
        for (int i = train_data->size - 1; i > 0; i--) {
            int j = (int)prng_below((u32)(i + 1));
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }

        for (int n = 0; n < train_data->size; n++) {
            int i = order[n];
            float *sample = train_data->images[i];

            /* Augmentation on roughly half the samples. A fully-connected net
               has no translation invariance, so shifting every image would
               train on a distribution the test set does not share. */
            if (prng_rand() & 1u) {
                int dx = (int)prng_below(3) - 1;
                int dy = (int)prng_below(3) - 1;
                shift_image(train_data->images[i], shifted, 28, 28, dx, dy);
                sample = shifted;
            }

            forward_propagate_with_activation(net, sample, output);

            /* One-hot target for the true digit. */
            int target[10] = {0};
            target[train_labels->labels[i]] = 1;

            total_loss += cross_entropy_loss(output, target, net->output_nodes);

            /* The same buffer must reach both passes: backpropagating against
               the unshifted image would compute gradients for another input. */
            backpropagation(net, sample, target, output, learning_rate);
        }

        double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("Epoch %d in %.2f s, Loss: %f\n", epoch + 1, secs, total_loss / train_data->size);
    }

    free(order);
    free(shifted);
    free(output);
}

/* ---------------- evaluation ---------------- */

float evaluate(neural_net *net, mnist_data *test_data, mnist_data *test_labels) {
    int correct_predictions = 0;

    /* Allocated once, outside the loop: the original re-allocated the same
       40-byte buffer ten thousand times. */
    float *output = (float *)malloc(net->output_nodes * sizeof(float));
    if (!output) {
        fprintf(stderr, "Failed to allocate the output buffer\n");
        return 0.0f;
    }

    for (int i = 0; i < test_data->size; i++) {
        forward_propagate_with_activation(net, test_data->images[i], output);

        /* argmax over the ten class probabilities */
        int predicted_label = 0;
        for (int j = 1; j < net->output_nodes; j++) {
            if (output[j] > output[predicted_label]) {
                predicted_label = j;
            }
        }

        /* Labels live in test_labels; test_data->labels was never initialised. */
        if (predicted_label == test_labels->labels[i]) {
            correct_predictions++;
        }
    }

    free(output);
    return (float)correct_predictions / test_data->size;
}

/* ---------------- serialisation ---------------- */

/* Layout: three layer sizes, then the raw weight and bias arrays. */
int save_network(neural_net *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Could not open %s\n", filename); return 1; }

    size_t n1 = (size_t)net->hidden_nodes * net->input_nodes;
    size_t n2 = (size_t)net->output_nodes * net->hidden_nodes;

    if (fwrite(&net->input_nodes,  sizeof(int), 1, f) != 1 ||
        fwrite(&net->hidden_nodes, sizeof(int), 1, f) != 1 ||
        fwrite(&net->output_nodes, sizeof(int), 1, f) != 1 ||
        fwrite(net->w1, sizeof(float), n1, f) != n1 ||
        fwrite(net->b1, sizeof(float), (size_t)net->hidden_nodes, f) != (size_t)net->hidden_nodes ||
        fwrite(net->w2, sizeof(float), n2, f) != n2 ||
        fwrite(net->b2, sizeof(float), (size_t)net->output_nodes, f) != (size_t)net->output_nodes) {
        fprintf(stderr, "Could not write %s\n", filename);
        fclose(f);
        return 1;
    }

    fclose(f);
    return 0;
}

int load_network(neural_net *net, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Could not open %s\n", filename); return 1; }

    int in_nodes = 0, hid_nodes = 0, out_nodes = 0;

    if (fread(&in_nodes,  sizeof(int), 1, f) != 1 ||
        fread(&hid_nodes, sizeof(int), 1, f) != 1 ||
        fread(&out_nodes, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Truncated header in %s\n", filename);
        fclose(f);
        return 1;
    }

    /* Refuse a file whose geometry does not match the network in memory:
       loading it would read past the end of the allocated arrays. */
    if (in_nodes  != net->input_nodes  ||
        hid_nodes != net->hidden_nodes ||
        out_nodes != net->output_nodes) {
        fprintf(stderr, "Shape mismatch: file %d-%d-%d, network %d-%d-%d\n",
                in_nodes, hid_nodes, out_nodes,
                net->input_nodes, net->hidden_nodes, net->output_nodes);
        fclose(f);
        return 1;
    }

    size_t n1 = (size_t)hid_nodes * in_nodes;
    size_t n2 = (size_t)out_nodes * hid_nodes;

    if (fread(net->w1, sizeof(float), n1, f) != n1 ||
        fread(net->b1, sizeof(float), (size_t)hid_nodes, f) != (size_t)hid_nodes ||
        fread(net->w2, sizeof(float), n2, f) != n2 ||
        fread(net->b2, sizeof(float), (size_t)out_nodes, f) != (size_t)out_nodes) {
        fprintf(stderr, "Incomplete data in %s\n", filename);
        fclose(f);
        return 1;
    }

    fclose(f);
    return 0;
}

/* ---------------- cleanup ---------------- */

void free_network(neural_net *net) {
    if (!net) return;
    free(net->w1);
    free(net->b1);
    free(net->w2);
    free(net->b2);
    free(net->hidden);
    free(net);
}

void free_mnist_images(mnist_data *data) {
    if (!data) return;
    if (data->images) {
        for (int i = 0; i < data->size; i++)
            free(data->images[i]);
        free(data->images);
    }
    free(data);
}

void free_mnist_labels(mnist_data *data) {
    if (!data) return;
    free(data->labels);
    free(data);
}

/* ---------------- main ---------------- */

int main(void) {
    /* Fixed seed on purpose: the accuracy quoted in the README is only
       reproducible if every run starts from the same weights and sees the
       samples in the same order. Swap in plat_get_entropy() to vary it. */
    prng_seed(20260822ULL, 54ULL);

    const char *train_image_filename = "dataset/train-images.idx3-ubyte";
    const char *train_label_filename = "dataset/train-labels.idx1-ubyte";
    const char *test_image_filename  = "dataset/t10k-images.idx3-ubyte";
    const char *test_label_filename  = "dataset/t10k-labels.idx1-ubyte";

    mnist_data *train_images = load_mnist_images(train_image_filename);
    mnist_data *train_labels = load_mnist_labels(train_label_filename);
    mnist_data *test_images  = load_mnist_images(test_image_filename);
    mnist_data *test_labels  = load_mnist_labels(test_label_filename);

    if (!train_images || !train_labels || !test_images || !test_labels) {
        printf("Failed to load images and labels.\n");
        /* Free whatever did load: a partial failure must not leak the rest. */
        free_mnist_images(train_images);
        free_mnist_labels(train_labels);
        free_mnist_images(test_images);
        free_mnist_labels(test_labels);
        return 1;
    }

    printf("Images and labels loaded successfully.\n");
    printf("Number of training images: %d\n", train_images->size);
    printf("Number of training labels: %d\n", train_labels->size);
    printf("Number of testing images: %d\n", test_images->size);
    printf("Number of testing labels: %d\n", test_labels->size);

    /* If the counts disagree, the two files are not a matching pair. */
    if (train_images->size != train_labels->size ||
        test_images->size  != test_labels->size) {
        fprintf(stderr, "Image and label counts do not match\n");
        free_mnist_images(train_images);
        free_mnist_labels(train_labels);
        free_mnist_images(test_images);
        free_mnist_labels(test_labels);
        return 1;
    }

    neural_net *net = initialize_network(784, 10, 128);
    if (!net) {
        fprintf(stderr, "Network initialisation failed\n");
        free_mnist_images(train_images);
        free_mnist_labels(train_labels);
        free_mnist_images(test_images);
        free_mnist_labels(test_labels);
        return 1;
    }

    const char *network_filename = "network.dat";
    if (load_network(net, network_filename) == 0) {
        printf("Loaded existing network from %s.\n", network_filename);
    } else {
        printf("No existing network found. Starting from scratch.\n");
    }

    /* 20 epochs: the loss flattens well before that, and 60,000 samples per
       epoch already means 1.2M weight updates. */
    train(net, train_images, train_labels, 20, 0.01f);

    if (save_network(net, network_filename) == 0) {
        printf("Network saved to %s.\n", network_filename);
    } else {
        printf("Failed to save the network.\n");
    }

    float accuracy = evaluate(net, test_images, test_labels);
    printf("Accuracy on test set: %.2f%%\n", accuracy * 100);

    free_network(net);
    free_mnist_images(train_images);
    free_mnist_labels(train_labels);
    free_mnist_images(test_images);
    free_mnist_labels(test_labels);

    return 0;
}