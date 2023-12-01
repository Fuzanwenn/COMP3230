/************************************************************
* Filename: llama2_3036174194.c
* Student name and Number: Fu Zanwen 3036174194
* Development platform: WSL Ubuntu 22.04, gcc version 11.2.0
* Remark: Complete all features
*************************************************************/

/*
PLEASE WRITE DOWN NAME AND UID BELOW BEFORE SUBMISSION
* NAME: Fu Zanwen
* UID : 3036174194

Please download the model and tokenizer to the same folder:
$ wget -O model.bin https://huggingface.co/huangs0/llama2.c/resolve/main/model.bin
$ wget -O tokenizer.bin https://huggingface.co/huangs0/llama2.c/resolve/main/tokenizer.bin

In compile, remember to add `-pthred` to link library:
$ gcc -o template template.c utilities.c -O2 -pthread -lm

Then Run with:
$ ./parallel
*/

#define _GNU_SOURCE // keep this line
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "utilities.h"
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>



/**
 * ----------------------------------------------------------------------------
 * TASK - Optimize Matrix-Vector Multiplication by Multi-Threading
 * 
 * Matrix-vector multiplication, used in QKV Mapping and Feed-Forward Network
 * is the most time-consuming part of GPT. Luckily, most of computation is 
 * independent of each other, so we can use parallel computing for acceleration.
 * 
 * Please use <pthread.h> and your favorite control method,
 * semaphore (please #include <semaphore.h>) / mutex lock + conditional variable
 * 
 * A sequential version is provided below, please modify it to parallel version.
*/

// YOUR CODE STARTS HERE

// additional header file

// Global structures and variables
struct rusage usage_main_thread;
struct rusage *usage_worker_threads;
pthread_t *worker_threads;
sem_t work_semaphore;

typedef struct {
    int begin;
    int end;
    float* output;
    const float* vector;
    const float* matrix;
    int columns;
} WorkerData;

WorkerData *data_for_workers;
int total_threads;

// Function to create matrix-vector multiplication threads
int create_mat_vec_mul(int thread_count) {
    total_threads = thread_count;
    if (total_threads > 0) {
        worker_threads = malloc(total_threads * sizeof(pthread_t));
        data_for_workers = malloc(total_threads * sizeof(WorkerData));
        usage_worker_threads = malloc(total_threads * sizeof(struct rusage));
        sem_init(&work_semaphore, 0, 0); // Initialize semaphore
    }
    return 0;
}

// Thread function
void *thread_function(void *arg) {
    WorkerData *data = (WorkerData *)arg;
    for (int i = data->begin; i < data->end; i++) {
        float accumulator = 0.0f;
        for (int j = 0; j < data->columns; j++) {
            accumulator += data->matrix[i * data->columns + j] * data->vector[j];
        }
        data->output[i] = accumulator;
    }
    sem_post(&work_semaphore); // Signal completion
    int idx = ((WorkerData *)arg) - data_for_workers;
    getrusage(RUSAGE_THREAD, &usage_worker_threads[idx]);
    return NULL;
}

// Function to perform matrix-vector multiplication
void mat_vec_mul(float* output, float* vector, float* matrix, int col, int row) {
    if (total_threads > 0) {
        int rows_per_worker = row / total_threads;
        for (int i = 0; i < total_threads; i++) {
            data_for_workers[i].begin = i * rows_per_worker;
            data_for_workers[i].end = (i == total_threads - 1) ? row : (i + 1) * rows_per_worker;
            data_for_workers[i].output = output;
            data_for_workers[i].vector = vector;
            data_for_workers[i].matrix = matrix;
            data_for_workers[i].columns = col;
            pthread_create(&worker_threads[i], NULL, thread_function, (void *)&data_for_workers[i]);
        }

        // Wait for all threads to signal completion
        for (int i = 0; i < total_threads; i++) {
            sem_wait(&work_semaphore);
            pthread_join(worker_threads[i], NULL);
        }
    } else {
        for (int i = 0; i < row; i++) {
            float accumulator = 0.0f;
            for (int j = 0; j < col; j++) {
                accumulator += matrix[i * col + j] * vector[j];
            }
            output[i] = accumulator;
        }
    }
}

// Function to clean up resources
int destroy_mat_vec_mul() {
    if (total_threads > 0) {
        free(worker_threads);
        free(data_for_workers);
        sem_destroy(&work_semaphore);
        free(usage_worker_threads);
    }
    getrusage(RUSAGE_SELF, &usage_main_thread);
    // Print resource usage
    printf("Main Thread - user: %.4f s, system: %.4f s\n",
           (usage_main_thread.ru_utime.tv_sec + usage_main_thread.ru_utime.tv_usec / 1000000.0),
           (usage_main_thread.ru_stime.tv_sec + usage_main_thread.ru_stime.tv_usec / 1000000.0));
    return 0;
}




// YOUR CODE ENDS HERE

int transformer(int token, int pos, LLMConfig* p, LLMRuntime* s, LLMWeight* w) {
    
    // a few convenience variables
    int dim = p->dim, hidden_dim =  p->hidden_dim, head_size = p->dim / p->n_heads;

    // copy the token embedding into x
    memcpy(s->x, &(w->token_embedding_table[token * dim]), dim*sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // Attention
        {
            // attention normalization
            normalize(s->xb, s->x, w->rms_att_weight + l*dim, dim);

            // q, k, v = w_q @ x, w_k @ x, w_v @ x, respectively
            mat_vec_mul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
            mat_vec_mul(s->k, s->xb, w->wk + l*dim*dim, dim, dim);
            mat_vec_mul(s->v, s->xb, w->wv + l*dim*dim, dim, dim);

            // apply positional embedding
            position_embedding(s->q, s->k, w, pos, p->dim, p->n_heads);

            // save intermediate result for later reference
            key_value_cache(l, pos, p, s);
            
            // attention calculation
            attention(l, pos, p, s, w);

            // wo @ x to get final result
            mat_vec_mul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

            // residual connection back into x
            accum(s->x, s->xb2, dim);
        }
    
        // Feed-Forward Network: w2 @ (silu(w1 @ x) * (w3 @ x)), * is element-wise multiply
        {
            // FFN Normalization
            normalize(s->xb, s->x, w->rms_ffn_weight + l*dim, dim);

            // w1 @ x
            mat_vec_mul(s->h1, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x)
            silu(s->h1, hidden_dim);
            // w3 @ x
            mat_vec_mul(s->h2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
            // silu(w1 @ x) * (w3 @ x)
            element_wise_mul(s->h1, s->h2, hidden_dim);
            // w2 @ (silu(w1 @ x) * (w3 @ x))
            mat_vec_mul(s->xb, s->h1, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

            // residual connection
            accum(s->x, s->xb, dim);
        }
    }
    
    // final normalization
    normalize(s->x, s->x, w->rms_final_weight, dim);
    // classifier into logits
    mat_vec_mul(s->logits, s->x, w->token_embedding_table, p->dim, p->vocab_size);
    // apply the temperature to the logits
    for (int q=0; q<p->vocab_size; q++) { s->logits[q] /= 0.9f; }
    // apply softmax to the logits to get the probabilities for next token
    softmax(s->logits, p->vocab_size);
    // now sample from this distribution to get the next token
    return sample(s->logits, p->vocab_size);
}

int main(int argc, char* argv[]) {

    unsigned int seed;
    int thr_count;

    if (argc == 3) {
        seed = atoi(argv[1]);
        thr_count = atoi(argv[2]);
    } else {
        printf("Usage: ./compiled <seed> <thr_count>\n");
        return 1;
    }

    // Initialize
    srand(seed);
    create_mat_vec_mul(thr_count);

    // load model
    LLMConfig config;
    LLMWeight weights;
    if (load_LLM_Config_Weight(&config, &weights) == 1) { return 1; }

    // load tokenizer
    char** vocab = malloc(config.vocab_size * sizeof(char*));
    if (load_tokenizer(vocab, config.vocab_size) == 1) { return 1; }

    // create and init the application LLMRuntime
    LLMRuntime state;
    malloc_LLMRuntime(&state, &config);
    
    // the current position we are in
    long start = time_in_ms();

    int next, token = 1, pos = 0; // token = 1 -> <START>
    while (pos < config.seq_len) {

        // forward the transformer to get logits for the next token
        next = transformer(token, pos, &config, &state, &weights);

        printf("%s", vocab[next]);
        fflush(stdout); // force print

        token = next;
        pos++;
    }

    long end = time_in_ms();
    printf("\n\nlength: %d, time: %f s, achieved tok/s: %f\n", config.seq_len, (double)(end-start)/1000, config.seq_len / (double)(end-start)*1000);

    // cleanup
    destroy_mat_vec_mul();
    free_LLMRuntime(&state);
    free_LLMWeight(&weights);
    for (int i = 0; i < config.vocab_size; i++) { free(vocab[i]); }
    free(vocab);
    return 0;
}