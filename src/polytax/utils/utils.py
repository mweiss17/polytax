def reduce_gradients(xla_found, is_multi_host, optimizer):
    # AllReduce the model gradients so we can step the global gradient
    if not xla_found:
        return
    import torch_xla.core.xla_model as xm

    xm.reduce_gradients(optimizer)

    if not is_multi_host:
        return

    # if self.is_master_ordinal:
    #     dist.all_reduce(param.grad.cpu() / dist.get_world_size())
    #     xm.all_reduce(xm.REDUCE_SUM, param.grad.to(self.device))
    # else:
    #     zeros = torch.zeros_like(param.grad)
    #     xm.all_reduce(xm.REDUCE_SUM, zeros)
