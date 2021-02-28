
#define KERNELS_COUNT 7
#define COUNT_OF_SUBMODELS 7
#define Total 900
#define STREAM_COUNT 7/*Theta*/
#define BINS_COUNT 2
enum RESIDUALS_1st{ R = 0, U = 1};
enum RESIDUALS_3st{ R_ = 2, U_ = 3 };
enum RESIDUALS_2x2{ Dh = 4, Dv = 5, Dd = 6};
enum MINMAX{ MAX = 1, MIN = 0};

extern uint3 blocks ;
extern uint3 threads;
const int HIST_COUNT = 7;

typedef struct PSRM_Features
{
	int index[HIST_COUNT];
	int submodel[HIST_COUNT];
	char name[HIST_COUNT][32];
	int* hists[HIST_COUNT];
	int sub_model_index[COUNT_OF_SUBMODELS];
	int sub_model_count[COUNT_OF_SUBMODELS];
	int last_index;
	int submodel_index;
}PSRM_Features;


#define PROJ_HIST_SPAM(host_residuals)\
 proj_hist_spam << <blocks, threads, 512, streams[host_features.last_index] >> >(host_residuals, dev_kernels, offsets,sizes, shifts, host_features.hists[host_features.last_index], q, shift_offset, src_width, src_height);



void make_models_1st(float ** host_dev_residuals, float* kernels, int*offsets, int* sizes, int* shifts, cudaStream_t streams[], PSRM_Features &host_features, int q, int src_width, int src_height);
void make_models_3st(float ** host_dev_residuals, float* kernels, int*offsets, int* sizes, int* shifts, cudaStream_t streams[], PSRM_Features &host_features, int q, int src_width, int src_height);
void make_models_2x2(float ** host_dev_residuals, float* kernels, int*offsets, int* sizes, int* shifts, cudaStream_t streams[], PSRM_Features &host_features, int q, int src_width, int src_height);

__global__ void proj_hist_spam(float* first_residual, float* kernels, int*offsets, int* sizes, int* shifts, int* out_hist, int q, int shift_offset, int src_width, int src_height);




