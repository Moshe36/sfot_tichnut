#include <iostream>
#include <fstream>
#include <streambuf>
#include <chrono>
#include <random>
#include <immintrin.h>
#include <thread>
#include <string>

using namespace std;

long Get_Time() {
    using chrono::high_resolution_clock;
    auto t = high_resolution_clock::now();
    auto nanosec = t.time_since_epoch();
    return nanosec.count() / 1000000;
}

class Matrix {
public:
    float* p;
    int rows;
    int cols;

    Matrix() : p(nullptr), rows(0), cols(0) {}
    Matrix(int rows_, int cols_) : p(new float[rows_ * cols_]), rows(rows_), cols(cols_) {}
    Matrix(int rows_, int cols_, float val) : p(new float[rows_ * cols_]), rows(rows_), cols(cols_) {
        for (int i = 0; i < rows * cols; i++)
            p[i] = val;
    }
    Matrix(int rows_, int cols_, float a, float b) : p(new float[rows_ * cols_]), rows(rows_), cols(cols_) { // Random matrix a(i,j) ~ U(a, b)
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(a, b);

        for (int i = 0; i < rows * cols; i++) p[i] = dis(gen);
    }
    Matrix(string path) {
        ifstream t(path);
        string str((istreambuf_iterator<char>(t)), istreambuf_iterator<char>());

        auto in_float = [](char ch) { return ('0' <= ch && ch <= '9') || (ch == '.'); };
        int rows_ = 0, cols_ = 0;

        for (int i = 0; i < str.size(); i++)
            if (str[i] == '\n') rows_++;

        for (int i1 = 0, i2 = 0; i2 < str.size() && str[i2] != '\n'; ) {
            for (i1 = i2; !in_float(str[i1]) && i1 < str.size(); i1++) {}
            for (i2 = i1; in_float(str[i2]) && i2 < str.size(); i2++) {}
            if (i1 != i2) cols_++;
        }

        rows = rows_; cols = cols_;
        p = new float[rows * cols];

        for (int i1 = 0, i2 = 0, j = 0; i2 < str.size(); ) {
            for (i1 = i2; !in_float(str[i1]) && i1 < str.size(); i1++) {}
            for (i2 = i1; in_float(str[i2]) && i2 < str.size(); i2++) {}
            if (i1 != i2) p[j++] = stof(str.substr(i1, i2 - i1));
        }
    }
    Matrix(Matrix& m) : p(new float[m.rows * m.cols]), rows(m.rows), cols(m.cols) {
        for (int i = 0; i < rows * cols; i++)
            p[i] = m.p[i];
    }
    Matrix(Matrix&& m) : rows(m.rows), cols(m.cols) {
        p = m.p;
        m.p = nullptr;
    }
    friend bool eq(Matrix& a, Matrix& b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix&& a, Matrix&& b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix&& a, Matrix& b) { return a.rows == b.rows && a.cols == b.cols; }
    friend bool eq(Matrix& a, Matrix&& b) { return a.rows == b.rows && a.cols == b.cols; }

    Matrix& operator = (Matrix& m) {
        if (p == m.p) return *this;
        if (eq(*this, m)) {
            for (int i = 0; i < rows * cols; i++)
                p[i] = m.p[i];
        }
        else {
            delete[] p;
            rows = m.rows;
            cols = m.cols;
            p = new float[rows * cols];
            for (int i = 0; i < rows * cols; i++)
                p[i] = m.p[i];
        }
        return *this;
    }
    Matrix& operator = (Matrix&& m) {
        if (p == m.p) return *this;
        p = m.p;
        m.p = nullptr;
        return *this;
    }
    ~Matrix() {
        if (p) delete[] p;
    }
    float& operator () (int i, int j) {
        if (0 <= i && i < rows && 0 <= j && j < cols)
            return p[i * cols + j];
        cerr << "Error of index in operator ()." << endl;
        return p[0];
    }
    /////////////////////////////// transpose ///////////////////////////////
    Matrix t() {
        Matrix tr(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                tr(j, i) = (*this)(i, j);
        return move(tr);
    }
    /////////////////////////////////// + ///////////////////////////////////
    friend Matrix operator + (Matrix& a, Matrix& b) {
        if (!eq(a, b)) {
            cerr << "Error of matrix size in operator +." << endl;
            return Matrix();
        }

        Matrix s(a.rows, a.cols);
        for (int i = 0; i < a.rows * a.cols; i++)
            s.p[i] = a.p[i] + b.p[i];
        return move(s);
    }
    friend Matrix operator + (Matrix&& a, Matrix&& b) {
        if (!eq(a, b)) {
            cerr << "Error of matrix size in operator +." << endl;
            return Matrix();
        }
        for (int i = 0; i < b.rows * b.cols; i++)
            b.p[i] += a.p[i];
        return move(b);
    }
    friend Matrix operator + (Matrix&& a, Matrix& b) { return move(b) + move(a); }
    friend Matrix operator + (Matrix& a, Matrix&& b) { return move(a) + move(b); }
    Matrix& operator += (Matrix&& m) {
        if (!eq(*this, m)) {
            cerr << "Error of matrix size in operator +=." << endl;
            return *this;
        }
        for (int i = 0; i < rows * cols; i++)
            p[i] += m.p[i];
        return *this;
    }
    Matrix& operator += (Matrix& m) { return operator+=(move(m)); }
    /////////////////////////////////// - ///////////////////////////////////
    friend Matrix operator - (Matrix& a, Matrix& b) {
        if (!eq(a, b)) {
            cerr << "Error of matrix size in operator -." << endl;
            return Matrix();
        }
        Matrix s(a.rows, a.cols);
        for (int i = 0; i < a.rows * a.cols; i++)
            s.p[i] = a.p[i] - b.p[i];
        return move(s);
    }
    friend Matrix operator - (Matrix&& a, Matrix&& b) {
        if (!eq(a, b)) {
            cerr << "Error of matrix size in operator -." << endl;
            return Matrix();
        }
        for (int i = 0; i < a.rows * a.cols; i++)
            a.p[i] -= b.p[i];
        return move(a);
    }
    friend Matrix operator - (Matrix&& a, Matrix& b) { return move(a) - move(b); }
    friend Matrix operator - (Matrix& a, Matrix&& b) {
        if (!eq(a, b)) {
            cerr << "Error of matrix size in operator -." << endl;
            return Matrix();
        }
        for (int i = 0; i < a.rows * a.cols; i++)
            b.p[i] = a.p[i] - b.p[i];
        return move(b);
    }
    Matrix& operator -= (Matrix&& m) {
        if (!eq(*this, m)) {
            cerr << "Error of matrix size in operator -=." << endl;
            return *this;
        }
        for (int i = 0; i < rows * cols; i++)
            p[i] -= m.p[i];
        return *this;
    }
    Matrix& operator -= (Matrix& m) { return operator-=(move(m)); }
    /////////////////////////////////// * ///////////////////////////////////
    friend Matrix operator * (Matrix& a, float b) {
        Matrix prod(a.rows, a.cols);
        for (int i = 0; i < a.rows * a.cols; i++)
            prod.p[i] = a.p[i] * b;
        return move(prod);
    }
    friend Matrix operator * (Matrix&& a, float b) {
        for (int i = 0; i < a.rows * a.cols; i++)
            a.p[i] *= b;
        return move(a);
    }
    Matrix& operator *= (Matrix&& m) {
        if (!eq(*this, m)) {
            cerr << "Error of matrix size in operator *." << endl;
            return *this;
        }
        for (int i = 0; i < rows * cols; i++)
            p[i] *= m.p[i];
        return *this;
    }
    Matrix& operator *= (Matrix& m) { return operator*=(move(m)); }

    friend Matrix operator * (float b, Matrix& a) { return a * b; }
    friend Matrix operator * (float b, Matrix&& a) { return a * b; }
    friend Matrix operator * (Matrix&& a, Matrix&& b) {
        if (a.cols != b.rows) {
            cerr << "Error of matrix size in operator *." << endl;
            return Matrix();
        }
        Matrix ret;
        ret.p = Tools::mult_thread_padd(a.rows, a.p, b.p, a.cols, b.cols, b.cols, Tools::dim_th, Tools::n_th);
        ret.rows = a.rows;
        ret.cols = b.cols;
        return ret;
    }
    friend Matrix operator * (Matrix& a, Matrix& b) { return move(a) * move(b); }
    friend Matrix operator * (Matrix&& a, Matrix& b) { return move(a) * move(b); }
    friend Matrix operator * (Matrix& a, Matrix&& b) { return move(a) * move(b); }

    friend ostream& operator << (ostream& out, Matrix&& m) {
        for (int i = 0; i < m.rows - 1; i++) {
            for (int j = 0; j < m.cols; j++)
                out << m(i, j) << "\t";
            out << endl;
        }
        for (int j = 0; j < m.cols; j++)
            out << m(m.rows - 1, j) << "\t";
        return out;
    }
    friend ostream& operator << (ostream& out, Matrix& m) {
        return out << move(m);
    }

    /////////////////////////////////// Matrix Operation Tools Class ///////////////////////////////////
    struct Tools {
        // Zero initialization of the block (m x n) in the matrix ("c" - start of the block, ldc - namber of colums in the matrix)
        static void init_c(int m, int n, float* c, int ldc)
        {
            for (int i = 0; i < m; i++, c += ldc)
                for (int j = 0; j < n; j += 8)
                    _mm256_storeu_ps(c + j, _mm256_setzero_ps());
        }

        // Multiplication of (6 x k) block of "a" and (k x 16) block of "b" ("b" - reordered) and streing it to (6 x 16) block in "c"
        static void kernel(int k, const float* a, const float* b, float* c, int lda, int ldb, int ldc)
        {
            __m256 a0, a1, b0, b1;

            __m256 c00 = _mm256_setzero_ps();    __m256 c01 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();    __m256 c11 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps();    __m256 c21 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps();    __m256 c31 = _mm256_setzero_ps();
            __m256 c40 = _mm256_setzero_ps();    __m256 c41 = _mm256_setzero_ps();
            __m256 c50 = _mm256_setzero_ps();    __m256 c51 = _mm256_setzero_ps();

            const int offset0 = lda * 0;         const int offset3 = lda * 3;
            const int offset1 = lda * 1;         const int offset4 = lda * 4;
            const int offset2 = lda * 2;         const int offset5 = lda * 5;

            for (int i = 0; i < k; i++)
            {
                b0 = _mm256_loadu_ps(b + 0);                  b1 = _mm256_loadu_ps(b + 8);

                a0 = _mm256_broadcast_ss(a + offset0);        a1 = _mm256_broadcast_ss(a + offset1);

                c00 = _mm256_fmadd_ps(a0, b0, c00);           c10 = _mm256_fmadd_ps(a1, b0, c10);
                c01 = _mm256_fmadd_ps(a0, b1, c01);           c11 = _mm256_fmadd_ps(a1, b1, c11);

                a0 = _mm256_broadcast_ss(a + offset2);        a1 = _mm256_broadcast_ss(a + offset3);

                c20 = _mm256_fmadd_ps(a0, b0, c20);           c30 = _mm256_fmadd_ps(a1, b0, c30);
                c21 = _mm256_fmadd_ps(a0, b1, c21);           c31 = _mm256_fmadd_ps(a1, b1, c31);

                a0 = _mm256_broadcast_ss(a + offset4);        a1 = _mm256_broadcast_ss(a + offset5);

                c40 = _mm256_fmadd_ps(a0, b0, c40);           c50 = _mm256_fmadd_ps(a1, b0, c50);
                c41 = _mm256_fmadd_ps(a0, b1, c41);           c51 = _mm256_fmadd_ps(a1, b1, c51);

                b += ldb; a++;
            }
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c00, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c01, _mm256_loadu_ps(c + 8)));
            c += ldc;
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c10, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c11, _mm256_loadu_ps(c + 8)));
            c += ldc;
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c20, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c21, _mm256_loadu_ps(c + 8)));
            c += ldc;
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c30, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c31, _mm256_loadu_ps(c + 8)));
            c += ldc;
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c40, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c41, _mm256_loadu_ps(c + 8)));
            c += ldc;
            _mm256_storeu_ps(c + 0, _mm256_add_ps(c50, _mm256_loadu_ps(c + 0)));
            _mm256_storeu_ps(c + 8, _mm256_add_ps(c51, _mm256_loadu_ps(c + 8)));
        }

        // Reordering of (k x 16) block of B
        static void reorder_b(int k, const float* b, int ldb, float* b_tmp)
        {
            for (int i = 0; i < k; i++, b += ldb, b_tmp += 16)
            {
                _mm256_storeu_ps(b_tmp + 0, _mm256_loadu_ps(b + 0));
                _mm256_storeu_ps(b_tmp + 8, _mm256_loadu_ps(b + 8));
            }
        }
        
        // Reordering of block of A
        static void reorder_a(const float* a, int mm, int kk, int lda, float* a_tmp){
            for(int i = 0; i < mm; i++)
                for(int j = 0; j < kk; j++)
                    a_tmp[i * kk + j] = a[i * lda + j];
        }
        
        //Localization
        static void macro(int m, int k, int n, const float* a, const float* b, float* c, float* b_tmp, int lda, int ldb, int ldc)
        {
            for (int j = 0; j < n; j += 16)
            {
                reorder_b(k, b + j, ldb, b_tmp);
                for (int i = 0; i < m; i += 6)
                    kernel(k, a + i * lda, b_tmp, c + i * ldc + j, lda, 16, ldc);
            }
        }
        
        // Product of matrices A (m x k) and B (k x n)
        static void mult(int m, int k, int n, const float* a, const float* b, float* c, int lda, int ldb, int ldc)
        {
            const int L1 = 32 * 1024, L2 = 256*1024;;
            int kk = std::min(L1 / 4 / 16, k);
            int mm = std::min(L2 / 4 / kk, m) / 6 * 6;
            float* b_tmp = new float[kk * 16];
            float* a_tmp = new float[kk * mm];
            
            for(int t = 0; t < k; t += kk)
            {
                int dk = std::min(k, t + kk) - t;
                for (int i = 0; i < m; i += mm)
                {
                    int dm = std::min(m, i + mm) - i;
                    if(t == 0)
                        init_c(dm, n, c + i * ldc, ldc);

                    reorder_a(a + i * lda + t, dm, dk, lda, a_tmp);
                    macro(dm, dk, n, a_tmp, b + t * ldb, c + i * ldc, b_tmp, dk, ldb, ldc);
                }
            }

            delete[] b_tmp;
            delete[] a_tmp;
        }

        // Multithreaded product of matrices A (m x k) and B (k x n)
        static float* mult_thread(int m, const float* a, const float* b, int lda, int ldb, int ldc, int dim_thread = dim_th, int n_thread = n_th) {
            int m_t;
            thread* t = new thread[n_thread];
            float* c = new float[m * ldc];

            switch (dim_thread) {
            case 0:
                m_t = m / n_thread;
                for (int i = 0; i < n_thread; i++)
                    t[i] = thread([&, i]() { mult(m_t, lda, ldc, a + i * m_t * lda, b, c + i * m_t * ldc, lda, ldb, ldc); });
                break;
            case 1:
                m_t = ldc / n_thread;
                for (int i = 0; i < n_thread; i++)
                    t[i] = thread([&, i]() { mult(m, lda, m_t, a, b + i * m_t, c + i * m_t, lda, ldb, ldc); });
                break;
            default:
                cerr << "Error in parametr 'dim_thread' in function 'mult_thread'." << endl;
                return nullptr;
            }
            for (int i = 0; i < n_thread; i++)
                t[i].join();

            return c;
        }

        static float* padd_mat(const float* a, int m, int n, int new_m, int new_n) {
            float* p = new float[new_m * new_n];
            int t = 0;

            for (int i = 0, j; i < m; i++) {
                for (j = 0; j < n; j++)
                    p[t++] = a[i * n + j];
                for (; j < new_n; j++)
                    p[t++] = 0;
            }

            for (; t < new_m * new_n; t++)
                p[t] = 0;

            return p;
        }

        static float* unpadd_mat(const float* a, int m, int n, int new_m, int new_n) {
            float* p = new float[new_m * new_n];

            for (int i = 0, j = 0, t = 0; i < new_m; i++, j += (n - new_n))
                for (int k = 0; k < new_n; k++, j++, t++)
                    p[t] = a[j];

            return p;
        }

        static float* mult_thread_padd(int m, const float* a, const float* b, int lda, int ldb, int ldc, int dim_thread = dim_th, int n_thread = n_th) {
            int c, m_new, lda_new, ldb_new, ldc_new;

            switch (dim_thread) {
            case 0:
                c = 6 * n_thread;
                lda_new = (lda % 16 == 0) ? lda : (lda / 16) * 16 + 16;
                ldb_new = (ldb % 16 == 0) ? ldb : (ldb / 16) * 16 + 16;
                ldc_new = (ldc % 16 == 0) ? ldc : (ldc / 16) * 16 + 16;
                m_new = (m % c == 0) ? m : (m / c) * c + c;
                break;
            case 1:
                c = 16 * n_thread;
                lda_new = (lda % 16 == 0) ? lda : (lda / 16) * 16 + 16;
                ldb_new = (ldb % c == 0) ? ldb : (ldb / c) * c + c;
                ldc_new = (ldc % c == 0) ? ldc : (ldc / c) * c + c;
                m_new = (m % 6 == 0) ? m : (m / 6) * 6 + 6;
                break;
            default:
                cerr << "Error in parametr 'dim_thread' in function 'mult_thread_padd'." << endl;
                return nullptr;
            }

            float* a_padd = nullptr, * b_padd = nullptr, * c_padd = nullptr, * ret = nullptr;
            bool is_a_padd = m_new != m || lda_new != lda;
            bool is_b_padd = lda_new != lda || ldb_new != ldb;

            if (is_a_padd) a_padd = padd_mat(a, m, lda, m_new, lda_new);
            if (is_b_padd) b_padd = padd_mat(b, lda, ldb, lda_new, ldb_new);

            if (is_a_padd && is_b_padd) {

                c_padd = mult_thread(m_new, a_padd, b_padd, lda_new, ldb_new, ldc_new, dim_thread, n_thread);

                ret = unpadd_mat(c_padd, m_new, ldc_new, m, ldc);
                delete[] a_padd;
                delete[] b_padd;
                delete[] c_padd;
            }
            if (is_a_padd && !is_b_padd) {
                c_padd = mult_thread(m_new, a_padd, b, lda_new, ldb_new, ldc_new, dim_thread, n_thread);

                ret = unpadd_mat(c_padd, m_new, ldc_new, m, ldc);
                delete[] a_padd;
                delete[] c_padd;
            }
            if (!is_a_padd && is_b_padd) {
                c_padd = mult_thread(m_new, a, b_padd, lda_new, ldb_new, ldc_new, dim_thread, n_thread);

                ret = unpadd_mat(c_padd, m_new, ldc_new, m, ldc);
                delete[] b_padd;
                delete[] c_padd;
            }
            if (!is_a_padd && !is_b_padd) {
                ret = mult_thread(m_new, a, b, lda_new, ldb_new, ldc_new, dim_thread, n_thread);
            }
            return ret;
        }
        static int n_th;
        static int dim_th;
    };
};
int Matrix::Tools::n_th = 8;
int Matrix::Tools::dim_th = 1;


float* mult(int M, int K, int N, float* A, float* B) {

    float* C = new float[M * N];
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0;

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
    return C;

}

int main(int argc, const char* argv[]) {
    long t1, t2;
    int m = 4000;
    int n = 4000;
    int k = 4000;

    Matrix a(m, k, 0, 1);
    Matrix b(k, n, 0, 1);

    auto x = b;
    t1 = Get_Time();
    Matrix ab = a * b;
    t2 = Get_Time();

    cout << "a * b time = " << t2 - t1 << endl;
    cout << ab(40, 120) << endl;

    return 0;
}
