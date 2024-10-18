import torch
from grouped_gemm.ops import permute, unpermute

indices = torch.tensor([[1, 3], [0, 3], [0, 1], [1, 3], [0, 1]], dtype=torch.int32, device='cuda')  # source_row 经过 topK 排序后，送给 ith expert
# source row 0 sent to expert 1, 3
# source row 1 sent to expert 0, 3
# source row 2 sent to expert 0, 2
# source row 3 sent to expert 1, 3
# source row 4 sent to expert 0, 1
input_act = torch.tensor([[0,0,0,0], [1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4]], dtype=torch.float32, device='cuda') # source_row，5 行
probs = torch.ones_like(indices, dtype=torch.float32)
permuted_inputs, row_id_map = permute(input_act, indices)
#unpermute_outputs = unpermute(permuted_inputs, row_id_map, probs)

print(row_id_map)
print(permuted_inputs)
