#include <fcntl.h>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <span>
#include <vector>
#include <string>
#include <stdexcept>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cctype>

// ----------------------------------------------------------------------------
// utilities: time
double g_matmul_ms = 0.0;
long g_matmul_calls = 0;


double time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1e6;
}

// ----------------------------------------------------------------------------
// Transformer model
struct GPTconfig {
    int n_embd;      // dim
    int hidden_dim;  // 4*n_embd
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_max;  // block_size


    int kv_dim() const {
        return n_embd * n_kv_heads / n_heads;
    }
};

struct TransformerWeights {
    std::span<const float> token_embedding_table;  // (vocab_size, n_embd)
    // weights for rmsnorms
    std::span<const float> rms_att_weight;    // (layer, n_embd)
    std::span<const float> rms_ffn_weight;    // (layer, n_embd)
    std::span<const float> rms_final_weight;  // (n_embd,)
    // weights for matmuls. note n_embd == n_heads * head_size
    std::span<const float> wq;  // (layer, n_embd, n_heads * head_size = n_embd)
    std::span<const float> wk;  // (layer, n_embd, n_kv_heads * head_size)
    std::span<const float> wv;  // (layer, n_embd, n_kv_heads * head_size)
    std::span<const float> wo;  // projection layer, (layer, n_heads * head_size, n_embd)
    // weights for ffn
    std::span<const float> w1;  // (layer, hidden_dim, n_embd)
    std::span<const float> w2;  // (layer, dim, n_embd)
    std::span<const float> w3;  // (layer, hidden_dim, n_embd)
    // (optional) classifier weights for the logits, on the last layer
    std::span<const float> wcls;
};

struct RunState {
    std::vector<float> x;            // (n_embd,)
    std::vector<float> x_temp;       // (n_embd,)
    std::vector<float> x_temp2;      // (n_embd,)
    std::vector<float> hid_temp;     // (hidden_dim,)
    std::vector<float> hid_temp2;    // (hidden_dim,)
    std::vector<float> q;            // query (dim,)
    std::vector<float> att;          // (n_heads, seq_max)
    std::vector<float> logits;       // output logits
    std::vector<float> key_cache;    // (layer, seq_len, kv_dim)
    std::vector<float> value_cache;  // (layer, seq_len, kv_dim)

    std::span<float> k;
    std::span<float> v;


    RunState() = default;

    explicit RunState(const GPTconfig& config) : 
        x(config.n_embd), 
        x_temp(config.n_embd),
        x_temp2(config.n_embd),
        hid_temp(config.hidden_dim),
        hid_temp2(config.hidden_dim),
        q(config.n_embd),
        att(config.n_heads * config.seq_max),
        logits(config.vocab_size),
        key_cache(config.n_layers * config.seq_max * config.kv_dim()),
        value_cache(config.n_layers * config.seq_max * config.kv_dim()) {}


    ~RunState() = default;
};

class MappedFile {
private:
    int fd_;
    size_t size_;
    void *data_;

public:
    explicit MappedFile(const std::string& checkpoint){
        fd_ = open(checkpoint.c_str(), O_RDONLY);
        if (fd_ == -1){
            throw std::runtime_error("Failed to open file: " + checkpoint);
        }

        struct stat st;
        if (fstat(fd_, &st) == -1){
            close(fd_);
            throw std::runtime_error("Failed to get file size");
        }
        size_ = st.st_size;

        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED){
            close(fd_);
            throw std::runtime_error("Failed to mmap file");
        }
    }

    ~MappedFile(){
        if (data_ != MAP_FAILED) munmap(data_, size_);
        if (fd_ != -1) close(fd_);
    };

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    MappedFile(MappedFile&& other) noexcept :
        fd_(other.fd_),
        size_(other.size_),
        data_(other.data_){
            other.fd_ = -1;
            other.data_ = MAP_FAILED;
        }

    size_t get_size() const { return size_; }
    
    void* get_data() const { return data_; }

    template <typename T>
    std::span<const T> get_span(size_t offset_elems, size_t count) const{
        const T* ptr = static_cast<const T*>(data_) + offset_elems;
        return std::span<const T>(ptr, count);
    }
};

class Transformer {
private:
    MappedFile mf_;
    GPTconfig config_;
    TransformerWeights weights_;
    RunState state_;


public:
    Transformer(const std::string &check_point) :
        mf_(check_point) {
            size_t offset_bytes = sizeof(GPTconfig);
            size_t offset_floats = offset_bytes / sizeof(float);

            if (mf_.get_size() < offset_bytes){
                throw std::runtime_error("File too small");
            }
            std::memcpy(&config_, mf_.get_data(), offset_bytes);
            
            int shared_weights = config_.vocab_size > 0? 1 : 0;
            config_.vocab_size = std::abs(config_.vocab_size);

            weights_.token_embedding_table = mf_.get_span<float>(
                offset_floats, config_.vocab_size * config_.n_embd);
            offset_floats += config_.vocab_size * config_.n_embd;
            weights_.rms_att_weight = mf_.get_span<float>(
                offset_floats, config_.n_layers * config_.n_embd);
            offset_floats += config_.n_layers * config_.n_embd;

            int head_size = config_.n_embd / config_.n_heads;
            weights_.wq = mf_.get_span<float>(
                offset_floats, config_.n_layers * config_.n_embd * config_.n_heads * head_size);
            offset_floats += config_.n_layers * config_.n_embd * config_.n_heads * head_size;
            weights_.wk = mf_.get_span<float>(
                offset_floats, config_.n_layers * config_.n_embd * config_.n_kv_heads * head_size);
            offset_floats += config_.n_layers * config_.n_embd * config_.n_kv_heads * head_size;
            weights_.wv = mf_.get_span<float>(
                offset_floats, config_.n_layers * config_.n_embd * config_.n_kv_heads * head_size);
            offset_floats += config_.n_layers * config_.n_embd * config_.n_kv_heads * head_size;
            weights_.wo = mf_.get_span<float>(
                offset_floats, config_.n_layers * config_.n_heads * head_size * config_.n_embd);
            offset_floats += config_.n_layers * config_.n_heads * head_size * config_.n_embd;
            
            weights_.rms_ffn_weight = mf_.get_span<float>(
                offset_floats, config_.n_layers * config_.n_embd);
            offset_floats += config_.n_layers * config_.n_embd;
            
            weights_.w1 = mf_.get_span<float>(
                offset_floats, config_.n_layers * config_.n_embd * config_.hidden_dim);
            offset_floats += config_.n_layers * config_.n_embd * config_.hidden_dim;
            weights_.w2 = mf_.get_span<float>(
                offset_floats, config_.n_layers * config_.hidden_dim * config_.n_embd);
            offset_floats += config_.n_layers * config_.hidden_dim * config_.n_embd;
            weights_.w3 = mf_.get_span<float>(
                offset_floats, config_.n_layers * config_.n_embd * config_.hidden_dim);
            offset_floats += config_.n_layers * config_.n_embd * config_.hidden_dim;
            
            weights_.rms_final_weight = mf_.get_span<float>(
                offset_floats, config_.n_embd);
            offset_floats += config_.n_embd;
            
            offset_floats += config_.seq_max * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
            offset_floats += config_.seq_max * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
            
            weights_.wcls = shared_weights ? 
                weights_.token_embedding_table : 
                mf_.get_span<float>(offset_floats, config_.vocab_size * config_.n_embd);
            state_ = RunState(config_);
        }

    ~Transformer() = default;

    GPTconfig get_config() const { return config_; }
    TransformerWeights get_weights() const { return weights_; }

    std::vector<float>& forward(int token, int pos);
};

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer
void rmsnorm(std::span<float> out, 
            std::span<const float> x, 
            std::span<const float> weight){
    // y_i = x_i * weight_i / sqrt( 1/d * Sigma x_i + Epsilon)
    float RMS = 0.0f;
    int size = x.size();

    for (int i = 0; i < size; i++){
        RMS += x[i] * x[i];
    }
    RMS /= size;
    RMS += 1e-5f;
    RMS = 1.0f / sqrtf(RMS);

    for (int i = 0; i < size; i++){
        out[i] = x[i] * RMS * weight[i];
    }
}

void softmax(std::span<float> x, int size){
    // find max value (for numerical stability)
    if (size == 0) return;
    float max_val = x[0];
    for (int i = 1; i < size; i++){
        if (max_val < x[i]){
            max_val = x[i];
        }
    }

    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++){
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    for (int i = 0; i < size; i++){
        x[i] /= sum;
    }
}

void matmul(std::span<float> out, std::span<const float> x, std::span<const float> weight, int n, int d){
    // W(d, n) @ x(n, ) -> xout(d, )
    double t0 = time_in_ms();
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++){
        float val = 0.0f;
        for (int j = 0; j < n; j++){
            val += weight[i * n + j] * x[j];
        }
        out[i] = val;
    }
    g_matmul_ms += time_in_ms() - t0;
    g_matmul_calls++;
}

std::vector<float>& Transformer::forward(int token, int pos){
    const GPTconfig& config = config_;
    const TransformerWeights& weights = weights_;
    RunState& state = state_;

    const int& n_embd = config.n_embd;
    const int& hidden_dim = config.hidden_dim;
    const int& head_size = n_embd / config.n_heads;
    const int& kv_dim = (n_embd * config.n_kv_heads) / config.n_heads;
    const int& kv_mul = config.n_heads / config.n_kv_heads; 

    auto x = std::span<float>(state.x);  // x (n_embd, )
    auto content_row = weights.token_embedding_table.subspan(n_embd*token, n_embd);
    std::memcpy(x.data(), content_row.data(), n_embd*sizeof(float));

    for (size_t l = 0; l < config.n_layers; l ++){
        
        // attention rmsnorm
        rmsnorm(state.x_temp, x, weights.rms_att_weight.subspan(l * n_embd, n_embd));


        // key and value point to the kv cache
        int loff = l * config.seq_max * kv_dim;
        state.k = std::span<float>(state.key_cache).subspan(loff + pos * kv_dim, kv_dim);
        state.v = std::span<float>(state.value_cache).subspan(loff + pos * kv_dim, kv_dim);

        matmul(state.q, state.x_temp, weights.wq.subspan(
            l * n_embd * n_embd, n_embd * n_embd), n_embd, n_embd);
        matmul(state.k, state.x_temp, weights.wk.subspan(
            l * n_embd * kv_dim, n_embd * kv_dim), n_embd, kv_dim);
        matmul(state.v, state.x_temp, weights.wv.subspan(
            l * n_embd * kv_dim, n_embd * kv_dim), n_embd, kv_dim);
        
        // rope position
        for (int i = 0; i < n_embd; i+=2){
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            
            for (int j = 0; j < rotn; j++){
                std::span<float> vec = j == 0? state.q : state.k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < config.n_heads; h++){
            auto q = std::span<float>(state.q).subspan(h * head_size, head_size);
            auto att = std::span<float>(state.att).subspan(h * config.seq_max, config.seq_max);

            for (int t = 0; t <= pos; t++){
                auto k = std::span<float>(state.key_cache).subspan(
                    loff + t * kv_dim + (h / kv_mul) * head_size, head_size);
                float score = 0.0f;
                for (int i = 0;i < head_size; i++){
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1); // (pos + 1, )

            auto xb = std::span<float>(state.x_temp).subspan(h * head_size, head_size);
            std::ranges::fill(xb, 0.0f);
            for (int t = 0; t <= pos; t++){
                auto v = std::span<float>(state.value_cache).subspan(
                    loff + t * kv_dim + (h / kv_mul) * head_size, head_size);
                float a = att[t];
                for (int i = 0; i < head_size; i++){
                    xb[i] += a * v[i];
                }
            }
        }

        matmul(state.x_temp2, state.x_temp, weights.wo.subspan(
            l * n_embd * n_embd, n_embd * n_embd), n_embd, n_embd);

        for (int i = 0; i < n_embd; i++){
            x[i] += state.x_temp2[i];
        }
        

        rmsnorm(state.x_temp, x, weights.rms_ffn_weight.subspan(l*n_embd, n_embd));

        matmul(state.hid_temp, state.x_temp, weights.w1.subspan(
            l * hidden_dim * n_embd, n_embd * hidden_dim), n_embd, hidden_dim);

        matmul(state.hid_temp2, state.x_temp, weights.w3.subspan(
            l * hidden_dim * n_embd, n_embd * hidden_dim), n_embd, hidden_dim);

        for (int i = 0; i < hidden_dim; i++){
            float val = state.hid_temp[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= state.hid_temp2[i];
            state.hid_temp[i] = val;
        }
        
        matmul(state.x_temp, state.hid_temp, weights.w2.subspan(
            l * n_embd * hidden_dim, hidden_dim * n_embd), hidden_dim, n_embd);

        for (int i = 0; i < n_embd; i++){
            x[i] += state.x_temp[i];
        }
    }

    rmsnorm(x, state.x, weights.rms_final_weight);

    matmul(state.logits, x, weights.wcls, n_embd, config.vocab_size);
    return state.logits;
}


// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
struct TokenIndex{
    std::string str;
    int id;

    bool operator<(const TokenIndex& other) const{
        return str < other.str;
    }
};

void safe_printf(const std::string& piece){
    if (piece[0] == '\0' ) { return ;}
    if (piece[1] == '\0' ){
        unsigned char byte_val = static_cast<unsigned char>(piece[0]);
        if (!(std::isprint(byte_val) || std::isspace(byte_val))) {  
            return; 
        }
    }
    printf("%s", reinterpret_cast<const char*>(piece.data()));
}

int str_lookup(const std::string& str, const std::vector<TokenIndex>& sorted_vocab){
    auto it = std::lower_bound(sorted_vocab.begin(), sorted_vocab.end(), str,
                    [](const TokenIndex& tok, const std::string& target){
                        return tok.str < target;
                    });
    if (it != sorted_vocab.end() && it->str == str){
        return it->id;
    }
    return -1;
}

class Tokenizer{
private:
    std::vector<std::string> vocab_;
    std::vector<float> vocab_scores_;
    std::vector<TokenIndex> sorted_vocab_;
    unsigned int max_token_length_;
    std::array<std::array<uint8_t, 2>, 256> byte_pieces_;

public:
    explicit Tokenizer(const std::string& tokenizer_path, int vocab_size){
        MappedFile mf(tokenizer_path);
        
        const uint8_t* ptr = static_cast<const uint8_t*>(mf.get_data());
        size_t offset = 0;
        
        std::memcpy(&max_token_length_, ptr + offset, sizeof(int));
        offset += sizeof(int);
        
        vocab_.resize(vocab_size);
        vocab_scores_.resize(vocab_size);
        sorted_vocab_.resize(vocab_size);

        for (int i = 0; i < 256; i++){
            byte_pieces_[i][0] = static_cast<uint8_t>(i);
            byte_pieces_[i][1] = '\0';
        }

        for (int i = 0; i < vocab_size; i++){
            std::memcpy(&vocab_scores_[i], ptr + offset, sizeof(float));
            offset += sizeof(float);

            int len;
            std::memcpy(&len, ptr + offset, sizeof(int));
            offset += sizeof(int);

            vocab_[i] = std::string(reinterpret_cast<const char*>(ptr + offset), len);
            offset += len;

            sorted_vocab_[i].str = vocab_[i];
            sorted_vocab_[i].id = i;
        }
        std::sort(sorted_vocab_.begin(), sorted_vocab_.end());
    }

    ~Tokenizer() = default;

    std::string decode(int prev_token, int token){
        std::string piece = vocab_[token];

        if (prev_token == 1 && piece[0] == ' ') { piece=piece.substr(1); }

        unsigned char byte_val;
        if (sscanf(piece.c_str(), "<0x%02hhX>", &byte_val) == 1){
            return std::string(reinterpret_cast<const char*>(byte_pieces_[byte_val].data()));
        }
        return piece;
    }

    void encode(const std::string& text, const int8_t& bos, const int8_t& eos, std::vector<int>& tokens){
        std::string str_buffer;
        tokens.clear();

        if (bos) { tokens.push_back(1); }

        if (!text.empty()){
            int dummy_prefix = str_lookup(" ", sorted_vocab_);
            tokens.push_back(dummy_prefix);
        }

        for (size_t i = 0; i < text.size(); i++){
            unsigned char c = text[i];
            if ((c & 0xC0) != 0x80) { str_buffer.clear(); }

            str_buffer +=c;

            if (i + 1 < text.size() && (static_cast<unsigned char>(text[i + 1]) & 0xC0) == 0x80 && str_buffer.size() < 4) { continue; }

            int id = str_lookup(str_buffer, sorted_vocab_);
            if (id != -1) {
                tokens.push_back(id);
            } else {
                for (char b: str_buffer) {
                    tokens.push_back(static_cast<unsigned char>(b) + 3);
                }
            }
            str_buffer.clear();
        }

        while (true){
            float best_score = -1e10;
            int best_id = -1;
            int best_idx = -1;

            for (size_t i = 0; i + 1< tokens.size(); i++){
                str_buffer = vocab_[tokens[i]] + vocab_[tokens[i + 1]];
                int id = str_lookup(str_buffer, sorted_vocab_);

                if (id != -1 && vocab_scores_[id] > best_score){
                    best_score = vocab_scores_[id];
                    best_id = id;
                    best_idx = i;
                }
            }

            if (best_idx == -1) { break; }

            tokens[best_idx] = best_id;

            tokens.erase(tokens.begin() + best_idx + 1);
        }

        if (eos) { tokens.push_back(2); }
    }
};
// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling
struct ProbIndex{
    float prob_;
    int index_;

    bool operator<(const ProbIndex& other) const{
        return prob_ > other.prob_;
    }
};

unsigned int random_u32(unsigned long long& state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long& state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

class Sampler{
private:
    int vocab_size_;
    std::vector<ProbIndex> probindex_;
    float temperature_;
    float topp_;
    unsigned long long rng_state_;

public:
    explicit Sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed) : 
        vocab_size_(vocab_size),
        temperature_(temperature),
        topp_(topp), 
        rng_state_(rng_seed) {
            probindex_.resize(vocab_size_);
        }

    ~Sampler() = default;

    int sample_mult(std::span<const float> prob, int n, float coin){
        float cdf = 0.0f;
        for (int i = 0; i < n; i++){
            cdf += prob[i];
            if (coin < cdf){
                return i;
            }
        }
        return n-1;
    }

    int sample_topp(std::span<const float> prob, int n, float topp, std::span<ProbIndex> probindex, float coin){
        int n0 = 0;
        const float cutoff = (1.0f - topp) / (n - 1);

        for (int i = 0; i < n; i++){
            if (prob[i] >= cutoff){
                probindex[n0].prob_ = prob[i];
                probindex[n0].index_ = i;
                n0++;
            }
        }
        std::sort(probindex.begin(), probindex.begin() + n0);

        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1;
        for (int i = 0; i < n0; i++){
            cumulative_prob += probindex[i].prob_;
            if (cumulative_prob > topp){
                last_idx = i;
                break;
            }
        }

        float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++){
            cdf += probindex[i].prob_;
            if (r < cdf){
                return probindex[i].index_;
            }
        }
        return probindex[last_idx].index_;
    }

    int sample(std::span<float> logits){
        int next;
        if (temperature_ == 0.0f) { 
            auto it = std::ranges::max_element(logits); 
            next = static_cast<int>(std::distance(logits.begin(), it));
        } else{
            for (int q = 0; q < vocab_size_; q++){ logits[q] /= temperature_; }
            softmax(logits, vocab_size_);
            float coin = random_f32(rng_state_);

            if (topp_ <= 0 || topp_ >= 1){
                next = sample_mult(logits, vocab_size_, coin);
            }else{
                next = sample_topp(logits, vocab_size_, topp_, probindex_, coin);
            }
        }
        return next;
    }
};


// ----------------------------------------------------------------------------
// generation loop
void generate(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler, std::string prompt, int steps){
    std::string empty_prompt;
    std::vector<int> prompt_tokens;

    tokenizer.encode(prompt, 1, 0, prompt_tokens);
    int num_prompt_tokens = prompt_tokens.size();

    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE); 
    }

    double start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps){
        auto& logits = transformer.forward(token, pos);

        if (pos < num_prompt_tokens - 1){
            next = prompt_tokens[pos + 1];
        }else{
            next = sampler.sample(logits);
        }
        pos++;

        if (next == 1) { break; }

        std::string piece = tokenizer.decode(token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); g_matmul_ms = 0; g_matmul_calls = 0; }
    }
    printf("\n");

    if (pos > 1){
        double end = time_in_ms();
        double total = (double)(end - start);
        fprintf(stderr, "achieved tok/s: %.4f\n", (pos-1) / total*1000);
        fprintf(stderr, "matmul: %.1f ms / %.1f ms = %.1f%%  (调用 %ld 次)\n",
            g_matmul_ms, total, g_matmul_ms / total * 100, g_matmul_calls);
    }
}

std::string read_stdin(const std::string& guide) {
    std::cout << guide << std::flush;
    std::string line;
    std::getline(std::cin, line);
    return line;
}
// ----------------------------------------------------------------------------
// chat loop
void chat(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
            std::string& cli_user_prompt, std::string& cli_system_prompt, int steps){
    std::string system_prompt;
    std::string user_prompt;
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    std::vector<int> prompt_tokens;
    int user_idx;

    int8_t user_turn = 1;
    int next = 0;
    int token = 0;
    int pos = 0;
    while (pos < steps) {
        if (user_turn) {
            prompt_tokens.clear();
            if (pos == 0) {
                if (cli_system_prompt.empty()) {
                    system_prompt = read_stdin("Enter system prompt (optional): ");
                } else {
                    system_prompt = cli_system_prompt;
                }
            }
            if (pos == 0 && !cli_user_prompt.empty()) {
                user_prompt = cli_user_prompt;
            } else {
                user_prompt = read_stdin("User: ");
            }
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt.c_str(), user_prompt.c_str());
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt.c_str());
            }
            tokenizer.encode(rendered_prompt, 1, 0, prompt_tokens);
            num_prompt_tokens = prompt_tokens.size();
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
        }

        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        }else{
            token = next;
        }
        if (token == 2) { user_turn = 1; }

        auto& logits = transformer.forward(token, pos);
        next = sampler.sample(logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            std::string piece = tokenizer.decode(token, next);
            safe_printf(piece);
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING


struct Args {
    std::string checkpoint_path;
    std::string tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    std::string prompt = "";
    unsigned int rng_seed = 0;
    std::string mode = "generate";
    std::string system_prompt = "";
};

void error_usage() {
    std::cerr << "Usage:   run <checkpoint> [options]\n"
              << "Example: run model.bin -n 256 -i \"Once upon a time\"\n"
              << "Options:\n"
              << "  -t <float>  temperature in [0,inf], default 1.0\n"
              << "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n"
              << "  -s <int>    random seed, default time(NULL)\n"
              << "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n"
              << "  -i <string> input prompt\n"
              << "  -z <string> optional path to custom tokenizer\n"
              << "  -m <string> mode: generate|chat, default: generate\n"
              << "  -y <string> (optional) system prompt in chat mode\n";
    std::exit(EXIT_FAILURE);
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    if (argc < 2) {
        error_usage();
    }
    args.checkpoint_path = argv[1];

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) {
            std::cerr << "Error: Missing value for flag " << argv[i] << "\n";
            error_usage();
        }
        std::string flag = argv[i];
        std::string val = argv[i + 1];

        if (flag.size() != 2 || flag[0] != '-') {
            std::cerr << "Error: Invalid flag format: " << flag << "\n";
            error_usage();
        }

        switch (flag[1]) {
            case 't': args.temperature = std::stof(val); break;
            case 'p': args.topp = std::stof(val); break;
            case 's': args.rng_seed = static_cast<unsigned int>(std::stoul(val)); break;
            case 'n': args.steps = std::stoi(val); break;
            case 'i': args.prompt = val; break;
            case 'z': args.tokenizer_path = val; break;
            case 'm': args.mode = val; break;
            case 'y': args.system_prompt = val; break;
            default:
                std::cerr << "Error: Unknown flag: " << flag << "\n";
                error_usage();
        }
    }

    if (args.rng_seed == 0) {
        args.rng_seed = static_cast<unsigned int>(std::time(nullptr));
    }
    if (args.temperature < 0.0f) args.temperature = 0.0f;
    if (args.topp <= 0.0f || args.topp > 1.0f) args.topp = 0.9f;
    if (args.steps < 0) args.steps = 0;

    return args;
}

// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    try{
        Args args = parse_args(argc, argv);

        std::cout << "Loading model from: " << args.checkpoint_path << std::endl;
        Transformer transformer(args.checkpoint_path);

        if (args.steps == 0 || args.steps > transformer.get_config().seq_max) {
                args.steps = transformer.get_config().seq_max;
        }

        Tokenizer tokenizer(args.tokenizer_path, transformer.get_config().vocab_size);
        Sampler sampler(transformer.get_config().vocab_size, args.temperature, args.topp, args.rng_seed);

        if (args.mode == "generate"){
            generate(transformer, tokenizer, sampler, args.prompt, args.steps);
        }else if (args.mode == "chat"){
            chat(transformer, tokenizer, sampler, args.prompt, args.system_prompt, args.steps);
        }else{
            fprintf(stderr, "unknown mode: %s\n", args.mode.c_str());
            error_usage();
        }
    } catch (const std::exception &e){
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return 0;
}
#endif