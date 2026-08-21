#include "main.h"

mnist_data* load_mnist_images(const char* images_filename) {
    FILE* fp = fopen(images_filename, "rb");
    if (!fp) {
        perror("fopen");
        return NULL;
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;

    fread(&magic_number, sizeof(magic_number), 1, fp);   
    magic_number = reverse_int(magic_number);

    if (magic_number != 2051) {
        fprintf(stderr, "Invalid MNIST images file: magic %d (atteso 2051)\n", magic_number);
        fclose(fp);                                      // FIX: era "file" + mancava il ;
        return NULL;                                     // FIX: mancava il ;
    }

    fread(&num_images, sizeof(num_images), 1, fp);      
    num_images = reverse_int(num_images);
    fread(&num_rows, sizeof(num_rows), 1, fp);           
    num_rows = reverse_int(num_rows);
    fread(&num_cols, sizeof(num_cols), 1, fp);           
    num_cols = reverse_int(num_cols);

    mnist_data* data = (mnist_data*)malloc(sizeof(mnist_data)); 
    if (!data) {
        fprintf(stderr, "Allocazione di mnist_data fallita\n");
        fclose(fp);
        return NULL;
    }

    data->size = num_images;
    data->images = (float**)calloc(num_images, sizeof(float*));
    if (!data->images) {
        fprintf(stderr, "Allocazione dell'array di puntatori fallita\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < num_images; i++) {
        data->images[i] = (float*)malloc(num_rows * num_cols * sizeof(float));

        if (!data->images[i]) {
            fprintf(stderr, "Allocazione fallita all'immagine %d di %d\n", i, num_images);
            for (int q = 0; q < i; q++) free(data->images[q]);
            free(data->images);
            free(data);
            fclose(fp);
            return NULL;
        }

        for (int j = 0; j < num_rows; j++) {
            for (int k = 0; k < num_cols; k++) {
                unsigned char temp = 0;
                fread(&temp, sizeof(temp), 1, fp);
            
                data->images[i][(num_cols * j) + k] = temp / 255.0f;  // normalizza in [0,1]
            }
        }
    }

    fclose(fp);
    return data;
}  

mnist_data* load_mnist_labels(const char* labels_filename) {
    FILE* fp = fopen(labels_filename, "rb");
    if (!fp) {
        perror("fopen");
        return NULL;
    }

    int magic_number = 0, num_labels = 0;

    fread(&magic_number, sizeof(magic_number), 1, fp);   
    magic_number = reverse_int(magic_number);

    if (magic_number != 2049) {
        fprintf(stderr, "Invalid MNIST labels file: magic %d (atteso 2049)\n", magic_number);
        fclose(fp);                                      // FIX: era "file" + mancava il ;
        return NULL;                                     // FIX: mancava il ;
    }

   
    fread(&num_labels, sizeof(num_labels), 1, fp);
    num_labels = reverse_int(num_labels);

    mnist_data* data = (mnist_data*)malloc(sizeof(mnist_data));  // FIX: era "minst_data"
    if (!data) {
        fprintf(stderr, "Allocazione di mnist_data fallita\n");
        fclose(fp);
        return NULL;
    }

    data->size = num_labels;
    data->labels = (int*)malloc(num_labels * sizeof(int));
    if (!data->labels) {
        fprintf(stderr, "Allocazione dell'array di etichette fallita\n");
        free(data);
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < num_labels; i++) {
        unsigned char temp = 0;
        fread(&temp, sizeof(temp), 1, fp);
        data->labels[i] = (int)temp;
    }

    fclose(fp);
    return data;
}   

neural_net* initialize_network(int input_nodes, int output_nodes, int hidden_nodes){
    neural_net *net = (neural_net*)malloc(sizeof(neural_net));
    if(net == NULL){
        perror("Memory allocation failed");
        return NULL;
    }

    net->input_nodes = input_nodes;
    net->output_nodes = output_nodes;
    net->hidden_nodes = hidden_nodes;

    net->w1     = (float*)malloc((size_t)hidden_nodes * input_nodes  * sizeof(float));
    net->b1     = (float*)calloc((size_t)hidden_nodes,                 sizeof(float));
    net->w2     = (float*)malloc((size_t)output_nodes * hidden_nodes * sizeof(float));
    net->b2     = (float*)calloc((size_t)output_nodes,                 sizeof(float));
    net->hidden = (float*)malloc((size_t)hidden_nodes                * sizeof(float));


    if (!net->w1 || !net->b1 || !net->w2 || !net->b2 || !net->hidden) {
        fprintf(stderr, "Allocazione dei parametri fallita\n");
        free_network(net);
        return NULL;
    }

    float limit1 = sqrtf(6.0f / (float)input_nodes);
    for (size_t i = 0; i < (size_t)hidden_nodes * input_nodes; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        net->w1[i] = (r * 2.0f - 1.0f) * limit1;
    }

    /* secondo strato: Xavier, perche' seguito da softmax */
    float limit2 = sqrtf(6.0f / (float)(hidden_nodes + output_nodes));
    for (size_t i = 0; i < (size_t)output_nodes * hidden_nodes; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        net->w2[i] = (r * 2.0f - 1.0f) * limit2;
    }
    
    return net; 
};

// This is the step where the input data (in this case, pixel values from the MNIST images) is passed through the neural network layers, 
// and each layer applies a set of weights and biases to produce an output. 
// This process allows the network to make predictions based on the input data.
void forward_propagation(neural_net *net, float *input, float *output){
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

//In the output layer, we use the softmax function to convert the raw scores (logits) into probabilities. 
// This is particularly useful for classification tasks where each class needs a probability score
void softmax(float *output, int size){
    float max = output[0]; 
    for(int i = 1; i < size; i++){
        if(output[i] > max){
            max = output[i]; 
        }
    }

    float sum = 0.0; 

    for(int i = 0; i < size; i++){
        output[i] = exp(output[i] - max); 
        sum += output[i]; 
    }

    for(int i = 0; i < size; i++){
        output[i] /= sum; 
    }
}

void forward_propagate_with_activation(neural_net *net, float *input, float *output){
    forward_propagation(net, input, output);
    softmax(output, net->output_nodes);
}

float cross_entropy_loss(float *output, int *target, int size){
    float loss = 0.0f; 
    for(int i = 0; i < size; i++){
        if(target[i] == 1){
            loss -= logf(fmaxf(output[i], 1e-7f));; 
        }
    }

    return loss; 
}

// Backpropagation: This is the algorithm used to train the neural network by updating the weights and biases to minimize the loss.
// - Gradient Descent: Backpropagation uses gradient descent to optimize the weights and biases. This involves calculating the gradient 
//   of the loss function with respect to each weight and bias, then updating them in the direction that reduces the loss.
//
// - Chain Rule: It applies the chain rule of calculus to compute these gradients efficiently. The chain rule helps in propagating 
//   the error back through the network, adjusting weights incrementally to minimize the error.
//
// - Learning Rate: Controls the size of the weight updates. A small learning rate ensures that the model converges smoothly to a minimum, 
//   while a large learning rate might speed up the process but risks overshooting the minimum.

void backpropagation(neural_net *net, float *input, int *target, float *output, float lr) {
    float delta2[16];
    float delta1[512];

    for (int o = 0; o < net->output_nodes; o++)
        delta2[o] = output[o] - target[o];

    /* --- delta1 usando i pesi VECCHI di w2 --- */
    for (int h = 0; h < net->hidden_nodes; h++)
        delta1[h] = 0.0f;

    for (int o = 0; o < net->output_nodes; o++) {
        const float d = delta2[o];
        const float *w = &net->w2[o * net->hidden_nodes];
        for (int h = 0; h < net->hidden_nodes; h++)
            delta1[h] += d * w[h];
    }

    for (int h = 0; h < net->hidden_nodes; h++)
        if (net->hidden[h] <= 0.0f) delta1[h] = 0.0f;   /* derivata ReLU */

    /* --- ora si può aggiornare --- */
    for (int o = 0; o < net->output_nodes; o++) {
        const float lrd = lr * delta2[o];
        float *w = &net->w2[o * net->hidden_nodes];
        for (int h = 0; h < net->hidden_nodes; h++)
            w[h] -= lrd * net->hidden[h];
        net->b2[o] -= lrd;
    }

    for (int h = 0; h < net->hidden_nodes; h++) {
        const float lrd = lr * delta1[h];
        if (lrd == 0.0f) continue;
        float *w = &net->w1[h * net->input_nodes];
        for (int in = 0; in < net->input_nodes; in++)
            w[in] -= lrd * input[in];
        net->b1[h] -= lrd;
    }
}

void shift_image(const float *src, float *dst, int rows, int cols, int dx, int dy) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int sr = r - dy, sc = c - dx;
            dst[r * cols + c] = (sr >= 0 && sr < rows && sc >= 0 && sc < cols) ? src[sr * cols + sc] : 0.0f;
        }
    }
}


void train(neural_net *net, mnist_data *train_data, mnist_data *train_labels, int epochs, float learning_rate) {
    float *output  = (float*)malloc(net->output_nodes * sizeof(float));
    float *shifted = (float*)malloc(28 * 28 * sizeof(float));
    int   *order   = (int*)malloc(train_data->size * sizeof(int));
    if (!output || !shifted || !order) { free(output); free(shifted); free(order); return; }

    for (int i = 0; i < train_data->size; i++) order[i] = i;

    for (int epoch = 0; epoch < epochs; epoch++) {
        clock_t t0 = clock();
        float total_loss = 0.0f;

        /* Fisher-Yates: ordine diverso a ogni epoca */
        for (int i = train_data->size - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
        }

        for (int n = 0; n < train_data->size; n++) {
            int i = order[n];
            float *sample = train_data->images[i];

            /* augmentation solo sul 50% dei campioni */
            if (rand() % 2) {
                int dx = (rand() % 3) - 1;
                int dy = (rand() % 3) - 1;
                shift_image(train_data->images[i], shifted, 28, 28, dx, dy);
                sample = shifted;
            }

            forward_propagate_with_activation(net, sample, output);

            int target[10] = {0};
            target[train_labels->labels[i]] = 1;

            total_loss += cross_entropy_loss(output, target, net->output_nodes);
            backpropagation(net, sample, target, output, learning_rate);
        }

        double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;
        printf("Epoch %d in %.2f s, Loss: %f\n", epoch + 1, secs, total_loss / train_data->size);
    }

    free(order);
    free(shifted);
    free(output);
}

/* ---------------- evaluate ---------------- */

float evaluate(neural_net *net, mnist_data *test_data, mnist_data *test_labels) {
    int correct_predictions = 0;

    // FIX: malloc spostata fuori dal ciclo. Prima allocava e liberava
    // 10000 volte lo stesso buffer da 40 byte, senza motivo.
    float *output = (float *)malloc(net->output_nodes * sizeof(float));
    if (!output) {                                  // FIX: controllo mancante
        fprintf(stderr, "Allocazione del buffer di output fallita\n");
        return 0.0f;
    }

    for (int i = 0; i < test_data->size; i++) {
        forward_propagate_with_activation(net, test_data->images[i], output);

        int predicted_label = 0;
        for (int j = 1; j < net->output_nodes; j++) {
            if (output[j] > output[predicted_label]) {
                predicted_label = j;
            }
        }

        // FIX: era test_data->labels[i]. Le etichette stanno in test_labels;
        // il campo labels di test_data non e' mai stato inizializzato e
        // contiene un puntatore casuale -> lettura di memoria non valida.
        if (predicted_label == test_labels->labels[i]) {
            correct_predictions++;
        }
    }

    free(output);
    return (float)correct_predictions / test_data->size;
}


/* ---------------- save_network ---------------- */
int save_network(neural_net *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Apertura di %s fallita\n", filename); return 1; }

    size_t n1 = (size_t)net->hidden_nodes * net->input_nodes;
    size_t n2 = (size_t)net->output_nodes * net->hidden_nodes;

    if (fwrite(&net->input_nodes,  sizeof(int), 1, f) != 1 ||
        fwrite(&net->hidden_nodes, sizeof(int), 1, f) != 1 ||
        fwrite(&net->output_nodes, sizeof(int), 1, f) != 1 ||
        fwrite(net->w1, sizeof(float), n1, f) != n1 ||
        fwrite(net->b1, sizeof(float), (size_t)net->hidden_nodes, f) != (size_t)net->hidden_nodes ||
        fwrite(net->w2, sizeof(float), n2, f) != n2 ||
        fwrite(net->b2, sizeof(float), (size_t)net->output_nodes, f) != (size_t)net->output_nodes) {
        fprintf(stderr, "Scrittura di %s fallita\n", filename);
        fclose(f);
        return 1;
    }

    fclose(f);
    return 0;
}

/* ---------------- load_network ---------------- */

int load_network(neural_net *net, const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Apertura di %s fallita\n", filename); return 1; }

    int in_nodes = 0, hid_nodes = 0, out_nodes = 0;


    if (fread(&in_nodes,  sizeof(int), 1, f) != 1 ||
        fread(&hid_nodes, sizeof(int), 1, f) != 1 ||
        fread(&out_nodes, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Header di %s troncato\n", filename);
        fclose(f);
        return 1;
    }

    if (in_nodes  != net->input_nodes  ||
        hid_nodes != net->hidden_nodes ||
        out_nodes != net->output_nodes) {
        fprintf(stderr, "Dimensioni incompatibili: file %d-%d-%d, rete %d-%d-%d\n",
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
        fprintf(stderr, "Dati di %s incompleti\n", filename);
        fclose(f);
        return 1;
    }

    fclose(f);
    return 0;
}


/* ---------------- free_network ---------------- */

void free_network(neural_net *net) {
    if (!net) return;
    free(net->w1);
    free(net->b1);
    free(net->w2);
    free(net->b2);
    free(net->hidden);
    free(net);
}


/* ---------------- funzioni di pulizia per i dati ---------------- */

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

int main(void) {                                    // FIX: (void), non ()
    srand((unsigned int)time(NULL));                // FIX: mancava del tutto.
                                                    // Senza, rand() parte sempre
                                                    // dallo stesso seme.

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
        // FIX: nell'originale, se il caricamento falliva a meta' si usciva
        // senza liberare i file gia' caricati. Ora libero sempre.
        free_mnist_images(train_images);
        free_mnist_labels(train_labels);
        free_mnist_images(test_images);
        free_mnist_labels(test_labels);
        return 1;                                   // FIX: exit code di errore
    }


    printf("Images and labels loaded successfully.\n");
    printf("Number of training images: %d\n", train_images->size);
    printf("Number of training labels: %d\n", train_labels->size);
    printf("Number of testing images: %d\n", test_images->size);
    printf("Number of testing labels: %d\n", test_labels->size);

    // FIX: controllo di coerenza. Se immagini ed etichette non
    // corrispondono in numero, i due file non sono la coppia giusta.
    if (train_images->size != train_labels->size ||
        test_images->size  != test_labels->size) {
        fprintf(stderr, "Immagini ed etichette non corrispondono\n");
        free_mnist_images(train_images);
        free_mnist_labels(train_labels);
        free_mnist_images(test_images);
        free_mnist_labels(test_labels);
        return 1;
    }

    neural_net *net = initialize_network(784, 10, 128);
    if (!net) {                                     // FIX: controllo mancante
        fprintf(stderr, "Inizializzazione della rete fallita\n");
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

    // FIX: 300 -> 20. Con 60000 immagini, 300 epoche sono 18 milioni di
    // aggiornamenti e ore di calcolo, mentre un modello a singolo strato
    // smette di migliorare molto prima.
    train(net, train_images, train_labels, 20, 0.01f);

    if (save_network(net, network_filename) == 0) {
        printf("Network saved to %s.\n", network_filename);
    } else {
        printf("Failed to save the network.\n");
    }

    float accuracy = evaluate(net, test_images, test_labels);  // FIX: era "Float"
    printf("Accuracy on test set: %.2f%%\n", accuracy * 100);

    free_network(net);
    free_mnist_images(train_images);
    free_mnist_labels(train_labels);
    free_mnist_images(test_images);
    free_mnist_labels(test_labels);

    return 0;
}