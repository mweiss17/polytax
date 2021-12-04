import torch
from torch import nn
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch_xla.core.functions import all_reduce, all_gather


class AllToAll(torch.autograd.Function):

  @staticmethod
  def forward(ctx, inputs, split_dimension, concat_dimension, split_count, groups=None):
    ctx.split_dimension = split_dimension
    ctx.concat_dimension = concat_dimension
    ctx.split_count = split_count
    ctx.groups = groups
    output = xm.all_to_all(inputs, split_dimension, concat_dimension, split_count, groups)
    #print(f"AllToAll forward. output.shape: {output.shape}")
    return output

  @staticmethod
  def backward(ctx, grad_outputs):
    #print(f"AllToAll backward.grad_outputs: {grad_outputs.shape}")
    return AllToAll.apply(grad_outputs, ctx.concat_dimension, ctx.split_dimension, ctx.split_count, ctx.groups), None, None, None, None

def all_to_all(input, split_dimension, concat_dimension, split_count, groups=None):
  """Performs an all-to-all distributed operation on the input tensor.
  This is the same as `xm.all_to_all()` but supports autograd differentiation.
  Args:
    input: A tensor of any dimension.
    split_dimension: The dimension to split the input tensor along.
    concat_dimension: The dimension to concatenate the output tensors along.
    split_count: The number of chunks to split the input tensor into.
    groups (list, optional): A list of list, representing the replica groups for
      the `all_to_all()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
  Returns:
    The reduced value across the selected replicas.
  """
  return AllToAll.apply(input, split_dimension, concat_dimension, split_count, groups)


def _mp_fn(index):

   print(f"index starting {index}")
   torch.manual_seed(index)
   n_tokens = 256
   num_cores = 8
   tokens_per_core = n_tokens // num_cores
   n_experts = 8
   expert_capacity = int(1.5 *  tokens_per_core * num_cores / n_experts)
   d_model = 4
   d_ff = 8

   inputs = (torch.rand([num_cores, tokens_per_core, d_model]) * index).float()
   expert_wi = torch.rand([n_experts, d_model, d_ff])
   expert_wo = torch.rand([n_experts, d_ff, d_model])
   router_linear = nn.Linear(d_model, n_experts)
   softmax = nn.Softmax(dim=-1)
   print(f"inputsshape: {inputs.shape}") 
   device = xm.xla_device()
   inputs = inputs.to(device)
   expert_wi = expert_wi.to(device)
   expert_wo = expert_wo.to(device)
   router_linear = router_linear.to(device)

   inputs.requires_grad = True
   expert_wi.requires_grad = True
   expert_wo.requires_grad = True
   router_linear.requires_grad = True

   router_logits = router_linear(inputs)
   print(f"router_logits: {router_logits.shape}")
   router_probs = softmax(router_logits)
   expert_gate, expert_index = torch.max(router_probs, dim=-1)
   expert_mask = torch.nn.functional.one_hot(expert_index, num_classes=n_experts)
   position_in_expert = torch.cumsum(expert_mask, dim=1) * expert_mask

   expert_mask *= torch.less(position_in_expert, expert_capacity + 1)
   expert_mask_flat = torch.sum(expert_mask, dim=-1)
   position_in_expert = torch.cumsum(expert_mask, dim=1) * expert_mask
   expert_gate *= expert_mask_flat

   combine_tensor = expert_gate.reshape(num_cores, tokens_per_core, 1, 1) * expert_mask_flat.reshape(num_cores, tokens_per_core, 1, 1) * F.one_hot(expert_index, num_classes=n_experts).unsqueeze(3) * F.one_hot(position_in_expert, num_classes=expert_capacity)
   dispatch_tensor = combine_tensor.bool()
   
   # (n_experts, n_cores, exper_capacity, d_model)
   expert_inputs = torch.einsum("btm,btxc->xbcm", inputs, dispatch_tensor.float())
   print(f"before all_to_all expert_inputs.shape: {expert_inputs.shape}") 
   expert_inputs = all_to_all(expert_inputs, 0, 0, split_count=num_cores)
   print(f"after all_to_all expert_inputs.shape: {expert_inputs.shape}, expert_wi: {expert_wi.shape}")
   layer1_out = torch.einsum('xmf,xbcm->xbcf', expert_wi, expert_inputs)
   expert_outputs = torch.einsum('xfm,xbcf->xbcm', expert_wo, layer1_out)
   final_output = torch.einsum('xbcm,btxc->btm', expert_outputs, combine_tensor.float())

   
   print(f"final_output.shape: {final_output.shape}")
   final_output = final_output.sum()
   
   final_output.backward()
   
   print(f"index: {index}, final_output: {final_output}, inputs.grad: {inputs.grad}")
    
   xm.rendezvous("finished")

if __name__ == "__main__":
    xmp.spawn(_mp_fn, args=(), nprocs=8, start_method="fork")

