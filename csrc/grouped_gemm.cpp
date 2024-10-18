#include "grouped_gemm.h"

#include <ATen/hip/HIPContext.h>
#include <c10/util/BFloat16.h>
#include <c10/hip/HIPStream.h>
#include <torch/extension.h>

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

namespace grouped_gemm {

#define NUM_STREAM 4

#define CUDA_CALL(code)					    \
  do {                                                      \
    hipError_t status = code;                              \
    std::string err = hipGetErrorString(status);           \
    TORCH_CHECK(status == hipSuccess, err);		    \
  } while (0)

#define CUBLAS_CALL(code)					  \
  do {								  \
    hipblasStatus_t status = code;				  \
    TORCH_CHECK(status == HIPBLAS_STATUS_SUCCESS, "CuBLAS Error"); \
  } while (0)

#define GROUPED_GEMM_STRINGIFY_HELPER(x) #x
#define GROUPED_GEMM_STRINGIFY(x) \
  GROUPED_GEMM_STRINGIFY_HELPER(x)

hipblasHandle_t cublas_handle[NUM_STREAM];
hipStream_t cublas_stream[NUM_STREAM];
hipEvent_t cublas_event[NUM_STREAM];
bool cublas_init = false;

void cublas_handle_init()
{
    cublas_init = true;

    for (int i = 0; i < NUM_STREAM; i++)
    {
        hipStreamCreateWithFlags(&cublas_stream[i], hipStreamNonBlocking);
        hipblasCreate(&cublas_handle[i]);
        hipblasSetStream(cublas_handle[i], cublas_stream[i]);
        hipEventCreate(&cublas_event[i]);
    }
}

inline void cublas_current_wait_streams(hipStream_t stream)
{
    for (int s = 0; s < NUM_STREAM; s++)
    {
        hipEventRecord(cublas_event[s], cublas_stream[s]);
    }

    for (int s = 0; s < NUM_STREAM; s++)
    {
        hipStreamWaitEvent(stream, cublas_event[s]);
    }
}

inline void cublas_streams_wait_current(hipStream_t stream)
{
    hipEventRecord(cublas_event[0], stream);

    for (int s = 0; s < NUM_STREAM; s++)
    {
        hipStreamWaitEvent(cublas_stream[s], cublas_event[0]);
    }
}

void CublasGemm(hipblasHandle_t cublas_handle,
    c10::BFloat16 *a, int64_t a_rows, int64_t a_cols, bool trans_a,
		c10::BFloat16 *b, int64_t b_rows, int64_t b_cols, bool trans_b,
		c10::BFloat16 *c, int64_t c_rows, int64_t c_cols) {
  int m = trans_b ? b_rows : b_cols;
  int k = trans_b ? b_cols : b_rows;
  int n = trans_a ? a_cols : a_rows;

  int lda = trans_a ? n : k;
  int ldb = trans_b ? k : m;
  hipblasOperation_t transpose_a = trans_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  hipblasOperation_t transpose_b = trans_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;


  float alpha = 1.0, beta = 0.0;
  CUBLAS_CALL(hipblasGemmEx_v2(cublas_handle,
			   transpose_b, transpose_a,
			   m, n, k, &alpha,
			   b, HIP_R_16BF, ldb,
			   a, HIP_R_16BF, lda,
			   &beta,
			   c, HIP_R_16BF, c_cols, HIP_R_32F,
			   HIPBLAS_GEMM_DEFAULT));
}

void CublasGroupedGemm(torch::Tensor a,
		       torch::Tensor b,
		       torch::Tensor c,
		       torch::Tensor batch_sizes,
		       bool trans_b) {
  if (!cublas_init)
    cublas_handle_init();

  int64_t bs = batch_sizes.size(0), k = a.size(1);
  int64_t n = trans_b ? b.size(1) : b.size(2);
  int64_t b_rows = b.size(1), b_cols = b.size(2);
  c10::BFloat16* a_ptr = a.data_ptr<c10::BFloat16>();
  c10::BFloat16* b_ptr = b.data_ptr<c10::BFloat16>();
  c10::BFloat16* c_ptr = c.data_ptr<c10::BFloat16>();

  cublas_streams_wait_current(c10::hip::getCurrentHIPStream());

  for (int i = 0; i < bs; ++i) {

    int64_t m = batch_sizes.data_ptr<int64_t>()[i];
    CublasGemm(cublas_handle[i % NUM_STREAM], a_ptr, m, k, /*trans_a=*/false,
	       b_ptr, b_rows, b_cols, trans_b,
	       c_ptr, m, n);
    a_ptr += m * k;
    b_ptr += b_rows * b_cols;
    c_ptr += m * n;
  }

  cublas_current_wait_streams(c10::hip::getCurrentHIPStream());
}

void CublasGroupedGemmVariableK(torch::Tensor a,
				torch::Tensor b,
				torch::Tensor c,
				torch::Tensor batch_sizes) {
  if (!cublas_init)
    cublas_handle_init();

  int64_t bs = batch_sizes.size(0), m = a.size(1), n = b.size(1);
  c10::BFloat16* a_ptr = a.data_ptr<c10::BFloat16>();
  c10::BFloat16* b_ptr = b.data_ptr<c10::BFloat16>();
  c10::BFloat16* c_ptr = c.data_ptr<c10::BFloat16>();

  cublas_streams_wait_current(c10::hip::getCurrentHIPStream());

  for (int i = 0; i < bs; ++i) {
    int64_t k = batch_sizes.data_ptr<int64_t>()[i];
    CublasGemm(cublas_handle[i % NUM_STREAM], a_ptr, k, m, /*trans_a=*/true,
	       b_ptr, k, n, /*trans_b=*/false,
	       c_ptr, m, n);
    a_ptr += k * m;
    b_ptr += k * n;
    c_ptr += m * n;
  }

  cublas_current_wait_streams(c10::hip::getCurrentHIPStream());
}

void GroupedGemmVariableK(torch::Tensor a,
			  torch::Tensor b,
			  torch::Tensor c,
			  torch::Tensor batch_sizes) {
  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_out) for 'b'.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 2);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);

  // Validate the dimensions.
  int64_t tokens = a.size(0), num_experts = batch_sizes.size(0);
  int64_t m = a.size(1), n = b.size(1);

  // Validate that we have the same contraction dimension.
  TORCH_CHECK(tokens == b.size(0));

  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 3);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.size(0) == num_experts);
  TORCH_CHECK(c.size(1) == m);
  TORCH_CHECK(c.size(2) == n);

  // Run the computation.
  CublasGroupedGemmVariableK(a, b, c, batch_sizes);
}

// NOTE: We only support dynamic group sizes for the 'a' tensor. Tensor 'b' is
// assumed to be batched with fixed sized batches.
//
// TODO(tgale): Validate alignment is true for every batch element.
void GroupedGemm(torch::Tensor a,
		 torch::Tensor b,
		 torch::Tensor c,
		 torch::Tensor batch_sizes,
		 bool trans_a, bool trans_b) {
  // NOTE: We only support 'trans_a' or 'trans_b', not both.
  TORCH_CHECK(!(trans_a && trans_b));

  // We expect the batch_sizes on CPU.
  TORCH_CHECK(batch_sizes.is_cpu());
  TORCH_CHECK(batch_sizes.ndimension() == 1);
  TORCH_CHECK(batch_sizes.scalar_type() == torch::kInt64);

  // We expected a CUDA tensor with two dimensions and shape
  // (tokens, hidden_in) for 'a'.
  TORCH_CHECK(a.is_cuda());
  TORCH_CHECK(a.ndimension() == 2);
  TORCH_CHECK(a.scalar_type() == torch::kBFloat16);

  // Defer to the variable 'k' helper for the rest of the op.
  if (trans_a) {
    GroupedGemmVariableK(a, b, c, batch_sizes);
    return;
  }

  // We expected a CUDA tensor with three dimensions and shape
  // (num_experts, hidden_in, hidden_out) for 'b'.
  TORCH_CHECK(b.is_cuda());
  TORCH_CHECK(b.ndimension() == 3);
  TORCH_CHECK(b.scalar_type() == torch::kBFloat16);

  // Validate the contraction dimensions match.
  int64_t tokens = a.size(0), num_experts = b.size(0);
  int64_t hidden_in = trans_b ? b.size(2) : b.size(1);
  int64_t hidden_out = trans_b ? b.size(1) : b.size(2);
  TORCH_CHECK(hidden_in == a.size(1));

  // Validate that we have one size per expert.
  TORCH_CHECK(batch_sizes.size(0) == num_experts);

  // Validate the output shape.
  TORCH_CHECK(c.is_cuda());
  TORCH_CHECK(c.ndimension() == 2);
  TORCH_CHECK(c.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(c.size(0) == tokens);
  TORCH_CHECK(c.size(1) == hidden_out);

  // NOTE: We support transposition through the 'trans_b' flag.
  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(b.is_contiguous());

  CublasGroupedGemm(a, b, c, batch_sizes, trans_b);
  return;
}

}  // namespace grouped_gemm
