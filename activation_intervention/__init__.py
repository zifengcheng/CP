from contextlib import contextmanager
from typing import Tuple, List, Callable, Optional
import torch as t
import torch.nn as nn


# Define types for pre-hook function, Hook, and Hooks
# 定义预处理钩子函数、钩子和钩子列表的类型

# PreHookFn_batch: Callable[[nn.Module, Tuple[t.Tensor]], t.Tensor]
# 预处理钩子函数类型 (处理批量数据)
#   - nn.Module: The module the hook is registered on. (钩子注册的模块)
#   - Tuple[t.Tensor]: The input to the module (as a tuple). (模块的输入，以元组形式)
#   - t.Tensor: The (potentially modified) input tensor. (可能被修改后的输入张量)
PreHookFn_batch = Callable[[nn.Module, Tuple[t.Tensor]], t.Tensor]

# PreHookFn: Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
# 预处理钩子函数类型 (类似于定义接口)
#   - nn.Module: The module the hook is registered on. (钩子注册的模块)
#   - t.Tensor: The input to the module. (模块的输入)
#   - Optional[t.Tensor]: The (potentially modified) input tensor, or None if no modification. (可能被修改后的输入张量，如果未修改则为 None)
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]] #类似定义接口

# Hook: Tuple[nn.Module, PreHookFn]
# 钩子类型：一个元组，包含一个模块和一个预处理钩子函数
#   - nn.Module: The module to hook. (要挂钩的模块)
#   - PreHookFn: The pre-hook function to be executed. (要执行的预处理钩子函数)
Hook = Tuple[nn.Module, PreHookFn]

# Hooks: List[Hook]
# 钩子列表类型：包含多个钩子的列表
Hooks = List[Hook]


@contextmanager
def pre_hooks(hooks: Hooks):
    """
    上下文管理器，用于在模块列表上注册前向预处理钩子。当上下文退出时，这些钩子将被移除。
    Context manager for registering forward pre-hooks on a list of modules.
    These hooks are removed when the context is exited.

    参数 (Args):
    - hooks (Hooks): 钩子列表，每个元素是一个 (模块, 钩子函数) 的元组。
                     List of hooks, where each element is a tuple of (module, hook_function).

    返回 (Yields):
    - List[torch.utils.hooks.RemovableHandle]: 钩子句柄列表，用于在退出上下文时移除钩子。
                                             List of hook handles, which can be used to remove the hooks.
    """
    try:
        # Register all pre-hooks and store their handles
        # 注册所有预处理钩子并存储它们的句柄
        handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
        yield handles # Yield the handles to the context
                      # 将句柄产出给上下文
    finally:
        # Ensure hooks are removed when exiting the context, even if errors occur
        # 确保在退出上下文时移除钩子，即使发生错误也是如此
        for handle in handles:
            handle.remove()


def get_blocks(model: nn.Module) -> nn.ModuleList:
    """
    获取模型中的 ModuleList。通常这代表模型中的主要重复构建块 (例如 Transformer 的层)。
    Retrieves a ModuleList from the model, typically representing the main repeating blocks (e.g., Transformer layers).

    参数 (Args):
    - model (nn.Module): 输入的模型。
                         The input model.

    返回 (Returns):
    - nn.ModuleList: 包含变压器块的 ModuleList。
                     A ModuleList containing the transformer blocks (or similar main blocks).
    """
    # Helper function to count the number of parameters in a module
    # 辅助函数，用于计算模块中的参数数量
    def numel_(mod):
        return sum(p.numel() for p in mod.parameters())

    model_numel = numel_(model) # Total number of parameters in the model (模型中的总参数数量)

    # Find ModuleList instances that contain a significant portion of the model's parameters
    # 查找包含模型参数一大部分的 ModuleList 实例
    candidates = [
        mod
        for mod in model.modules() # Iterate over all modules in the model (遍历模型中的所有模块)
        if isinstance(mod, nn.ModuleList) and numel_(mod) > 0.5 * model_numel # Check if it's a ModuleList and has >50% of model parameters (检查是否为 ModuleList 并且参数数量超过模型总参数的50%)
    ]
    # Ensure exactly one such ModuleList is found
    # 确保只找到一个这样的 ModuleList
    assert len(candidates) == 1, f'Found {len(candidates)} ModuleLists with >50% of model params.'
    return candidates[0] # Return the identified ModuleList (返回找到的 ModuleList)







def get_activation_modification_hook(
    negative_activations_batch: t.Tensor,
    coefficient: float,
    modification_strategy: str,
    num_heads: int,
    head_dim: int,
    meta_indice : int,
    meta_prompts_size: int
) -> Callable[[nn.Module, Tuple[t.Tensor]], Optional[t.Tensor]]:
    """
    返回用于激活值的钩子函数。此钩子根据指定的策略修改模块的输入激活。
    Returns a hook function for modifying activation values.
    This hook modifies the input activations of a module based on a specified strategy.

    干预策略 (Intervention Strategies):
    - norm: A_new = (A - B) * ||A|| / ||A - B||
    - scaled: A_new = C * (A - B)
    - scaled_norm: A_new = C * (A - B) * ||A|| / ||A - B||
    - sub_head_norm: Normalizes (A - B) per head, maintaining original per-head norm, then combines. (按注意力头归一化 (A-B)，保持原始的每个头的范数，然后合并)
    - self_scaled: A_new = A * C (对自身激活进行缩放)

    参数 (Args):
    - negative_activations_batch (t.Tensor): “负面”激活批次 (B)，用作修改的基准。
                                             Batch of "negative" activations (B), used as a baseline for modification.
    - coefficient (float): 修改策略中使用的系数 (C)。
                           Coefficient (C) used in the modification strategy.
    - modification_strategy (str): 要应用的修改策略的名称。
                                   Name of the modification strategy to apply.
    - num_heads (int): 注意力头的数量 (用于 'sub_head_norm')。
                       Number of attention heads (used for 'sub_head_norm').
    - head_dim (int): 每个注意力头的维度 (用于 'sub_head_norm')。
                      Dimension of each attention head (used for 'sub_head_norm').
    - meta_indice (int): 元提示中要进行干预的索引上界（不包含）。
                         Upper bound (exclusive) for indices in meta prompts to be intervened on.
    - meta_prompts_size (int): 元提示的总大小，用于确定批处理中元提示的边界。
                               Total size of meta prompts, used to determine boundaries of meta prompts in a batch.

    返回 (Returns):
    - Callable[[nn.Module, Tuple[t.Tensor]], Optional[t.Tensor]]: 可以注册为前向预处理钩子的函数。
                                                                A function that can be registered as a forward pre-hook.
    """

    # ---------------------------
    # 策略实现函数 (Strategy Implementation Functions)
    # ---------------------------
    
    def _normalized_difference_hook(_: nn.Module, inputs: Tuple[t.Tensor]) -> t.Tensor:
        """
        实现 A_new = (A-B) * ||A|| / ||A-B|| 的修改策略。
        Implements the modification strategy: A_new = (A-B) * ||A|| / ||A-B||.
        A: residual_input (当前激活)
        B: negative_activations_batch (负面激活)
        修改后的激活将具有与 (A-B) 相同的方向，但其范数被调整为与原始 A 的范数相同。
        The modified activation will have the same direction as (A-B), but its norm is scaled to match the norm of the original A.
        """
        residual_input, = inputs # Unpack the input tuple (解包输入元组)

        # meta_indice == 2 (或其他表示“全部干预”的值) 或 meta_indice >= meta_prompts_size 时，对整个批次进行干预
        # If meta_indice indicates "intervene all" (e.g., 7) or covers all meta prompts, intervene on the whole batch.
        if meta_indice != 2 and meta_indice < meta_prompts_size: # Check if only a subset of meta prompts needs intervention (检查是否只有一部分元提示需要干预)
            # 干预部分meta (Intervene on a part of the meta prompts)
            intervention_indices = [] # Store indices for intervention (存储需要干预的索引)
            common_indices = []       # Store indices for common (non-intervened) part (存储普通（未干预）部分的索引)

            # Iterate through the batch in chunks of meta_prompts_size
            # 以 meta_prompts_size 为块大小遍历批次
            for i in range(0, residual_input.shape[0], meta_prompts_size):
                intervention_indices.extend(range(i, i + meta_indice))  # Select the first 'meta_indice' samples in the chunk for intervention (选择块中前 'meta_indice' 个样本进行干预)
                common_indices.extend(range(i + meta_indice, i + meta_prompts_size)) # The rest are common (其余为普通样本)

            # Separate the input into intervention and common parts
            # 将输入分为干预部分和普通部分
            meta_residual = residual_input[intervention_indices]
            negative_activations = negative_activations_batch[intervention_indices] # Select corresponding negative activations (选择相应的负面激活)
            common_residual = residual_input[common_indices]

            # Calculate the original norm of the last token's activation for the intervention part
            # 计算干预部分最后一个token激活的原始范数
            original_norm = meta_residual[:, -1:, :].norm(dim=2, keepdim=True)

            # 计算差异并归一化 (Calculate the difference)
            # A-B for the intervention part
            activation_diff = meta_residual[:, -1:, :] - negative_activations[:, -1:, :]
            # A_new direction is A-B
            modified_activations = activation_diff

            # 保持原始范数 (Maintain original norm)
            # Calculate the norm of the difference vector (计算差分向量的范数)
            new_norm = modified_activations.norm(dim=2, keepdim=True)
            # Avoid division by zero if new_norm is very small
            # 如果 new_norm 非常小，避免除以零
            scaling_factor = original_norm / (new_norm + 1e-7) # Add epsilon for numerical stability (增加 epsilon 以保证数值稳定性)
            # Apply scaling: (A-B) * ||A|| / ||A-B||
            meta_residual[:, -1:, :] = modified_activations * scaling_factor
            # Concatenate the modified intervention part and the original common part
            # 合并修改后的干预部分和原始的普通部分
            residual_input = t.cat([meta_residual, common_residual], dim=0)
        else:
            # Intervene on the entire batch (对整个批次进行干预)
            # Calculate the original norm of the last token's activation
            # 计算最后一个token激活的原始范数
            original_norm = residual_input[:, -1:, :].norm(dim=2, keepdim=True)

            # 计算差异 (Calculate the difference A-B)
            activation_diff = residual_input[:, -1:, :] - negative_activations_batch[:, -1:, :]

            # 保持原始范数 (Maintain original norm)
            # Calculate the norm of the difference vector (计算差分向量的范数)
            new_norm = activation_diff.norm(dim=2, keepdim=True)
            # Avoid division by zero
            # 避免除以零
            scaling_factor = original_norm / (new_norm + 1e-7) # Add epsilon (增加 epsilon)
            # Apply scaling: (A-B) * ||A|| / ||A-B||
            residual_input[:, -1:, :] = activation_diff * scaling_factor
        return residual_input



    def _scaled_difference_hook(_: nn.Module, inputs: Tuple[t.Tensor]) -> t.Tensor:
        """
        实现 A_new = C*(A-B) 的修改策略。
        Implements the modification strategy: A_new = C*(A-B).
        A: residual_input (当前激活)
        B: negative_activations_batch (负面激活)
        C: coefficient (系数)
        修改后的激活是 (A-B) 方向上的向量，其大小由系数 C 缩放。
        The modified activation is a vector in the direction of (A-B), scaled by the coefficient C.
        """
        residual_input, = inputs # Unpack the input tuple (解包输入元组)
        # Calculate the difference A-B (计算差异 A-B)
        activation_diff = residual_input[:, -1:, :] - negative_activations_batch[:, -1:, :]
        # Scale the difference by the coefficient C (用系数 C 缩放差异)
        residual_input[:, -1:, :] = coefficient * activation_diff
        return residual_input

    def _normalized_scaled_difference_hook(_: nn.Module, inputs: Tuple[t.Tensor]) -> t.Tensor:
        """
        实现 A_new = C * (A-B) * ||A|| / ||A-B|| 的修改策略。
        Implements the modification strategy: A_new = C * (A-B) * ||A|| / ||A-B||.
        A: residual_input (当前激活)
        B: negative_activations_batch (负面激活)
        C: coefficient (系数)
        修改后的激活方向为 (A-B)，其范数被调整为与原始 A 的范数成比例（由 C 缩放）。
        The modified activation has the direction of (A-B), and its norm is scaled proportionally to the original A's norm (scaled by C).
        """
        residual_input, = inputs # Unpack the input tuple (解包输入元组)
        # Calculate the original norm of the last token's activation
        # 计算最后一个token激活的原始范数
        original_norm = residual_input[:, -1:, :].norm(dim=2, keepdim=True)

        # Calculate the difference A-B (计算差异 A-B)
        activation_diff = residual_input[:, -1:, :] - negative_activations_batch[:, -1:, :]

        # 计算归一化因子 (Calculate the normalization factor)
        # Norm of the difference vector (差分向量的范数)
        new_norm = activation_diff.norm(dim=2, keepdim=True)
        # Avoid division by zero
        # 避免除以零
        scaling_factor = original_norm / (new_norm + 1e-7) # Add epsilon (增加 epsilon)
        # Apply scaling: C * (A-B) * (||A|| / ||A-B||)
        residual_input[:, -1:, :] = coefficient * activation_diff * scaling_factor
        return residual_input

    def _multihead_normalized_hook(module: nn.Module, inputs: Tuple[t.Tensor]) -> t.Tensor:
        """
        多头注意力机制下的归一化差异处理：对每个子头单独计算 norm 后再合并。
        Normalized difference processing under multi-head attention mechanism:
        Calculate norm for each sub-head separately, then merge.
        A_new_head_i = (A_head_i - B_head_i) * ||A_head_i|| / ||A_head_i - B_head_i||
        A: residual_input (当前激活)
        B: negative_activations_batch (负面激活)
        """
        residual_input, = inputs # Unpack the input tuple (解包输入元组)
        batch_size, base_q_len, _ = residual_input.shape # Get dimensions (获取维度)

        # 转换为多头表示，形状变为 (batch_size, num_heads, base_q_len, head_dim)
        # Reshape to multi-head representation: (batch_size, num_heads, base_q_len, head_dim)
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        multihead_input = residual_input.view(batch_size, base_q_len, num_heads, head_dim).transpose(1, 2)

        # 取出每个子头在最后一个位置的表示，形状为 (batch_size, num_heads, head_dim)
        # Get the representation of the last token for each head: (batch_size, num_heads, head_dim)
        multihead_last = multihead_input[:, :, -1, :] # This is A_head_i for the last token (这是最后一个token的 A_head_i)

        # 分别计算每个子头的原始范数（对 head_dim 维度计算 norm），结果形状 (batch_size, num_heads, 1)
        # Calculate the original norm for each head (norm over head_dim): (batch_size, num_heads, 1)
        # This is ||A_head_i|| for the last token (这是最后一个token的 ||A_head_i||)
        original_norm = multihead_last.norm(dim=-1, keepdim=True)

        # 对负样本张量 negative_activations_batch 也做相同处理
        # Perform the same reshaping for the negative_activations_batch tensor
        act_q_len = negative_activations_batch.shape[1] # Sequence length of negative activations (负面激活的序列长度)
        # (batch, act_q_len, hidden_dim) -> (batch, act_q_len, num_heads, head_dim) -> (batch, num_heads, act_q_len, head_dim)
        negative_multihead = negative_activations_batch.view(batch_size, act_q_len, num_heads, head_dim).transpose(1, 2)
        # Get the last token representation for negative activations: (batch_size, num_heads, head_dim)
        # 获取负面激活的最后一个token表示
        negative_last = negative_multihead[:, :, -1, :] # This is B_head_i for the last token (这是最后一个token的 B_head_i)

        # 计算每个子头 (A_head_i - B_head_i) 的范数，形状 (batch_size, num_heads, 1)
        # Calculate the norm of (A_head_i - B_head_i) for each head: (batch_size, num_heads, 1)
        # This is ||A_head_i - B_head_i|| for the last token (这是最后一个token的 ||A_head_i - B_head_i||)
        diff_norm = (multihead_last - negative_last).norm(dim=-1, keepdim=True)

        # Calculate scaling factor: ||A_head_i|| / ||A_head_i - B_head_i||
        # 计算缩放因子
        scaling_factor = original_norm / (diff_norm + 1e-7) # Add epsilon (增加 epsilon) # Shape: (batch_size, num_heads, 1)

        # 用各自的缩放因子对每个子头在最后位置的差异进行归一化
        # Normalize the difference for each head at the last position using its respective scaling factor
        # New A_head_i = (A_head_i - B_head_i) * scaling_factor
        multihead_input[:, :, -1, :] = (multihead_last - negative_last) * scaling_factor

        # 合并多头表示，恢复到原始形状 (batch_size, base_q_len, hidden_size)
        # Merge multi-head representations back to original shape: (batch_size, base_q_len, hidden_size)
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, hidden_dim)
        return multihead_input.transpose(1, 2).reshape(batch_size, base_q_len, -1)

    def _scaled_self_hook(module: nn.Module, inputs: Tuple[t.Tensor]) -> t.Tensor:
        """
        实现 A_new = A * C 的修改策略 (直接缩放自身激活)。
        Implements the modification strategy: A_new = A * C (directly scale self activation).
        A: residual_input (当前激活)
        C: coefficient (系数)
        """
        output,  = inputs # Unpack the input tuple (解包输入元组)
        return output * coefficient # Scale the input by the coefficient (用系数缩放输入)

    # ---------------------------
    # 策略选择逻辑 (Strategy Selection Logic)
    # ---------------------------
    # Map strategy names to their corresponding hook functions
    # 将策略名称映射到其相应的钩子函数
    strategy_mapping = {
        "scaled": _scaled_difference_hook,
        "norm": _normalized_difference_hook,
        "scaled_norm": _normalized_scaled_difference_hook,
        "sub_head_norm": _multihead_normalized_hook,
        "self_scaled": _scaled_self_hook
    }

    # Return the selected hook function based on modification_strategy
    # 根据 modification_strategy 返回选定的钩子函数
    selected_hook = strategy_mapping.get(modification_strategy)
    if selected_hook is None:
        raise ValueError(f"Unknown modification strategy: {modification_strategy}. Available strategies: {list(strategy_mapping.keys())}")
    return selected_hook



def get_o_proj_input(
    model: nn.Module, tokenizer, prompts: List[str], layer: int
) -> t.Tensor:
    """
    获取指定层的自注意力模块中线性层 o_proj 的输入。
    Gets the input to the o_proj linear layer within the self-attention module of a specified layer.

    参数 (Args):
    - model (nn.Module): 输入的模型。
                         The input model.
    - tokenizer: 分词器。
                 The tokenizer.
    - prompts (List[str]): 提示词列表。
                           List of prompt strings.
    - layer (int): 指定的层索引。
                   The specified layer index.

    返回 (Returns):
    - t.Tensor: 指定层的自注意力模块通过线性层 o_proj 的输入。
                The input to the o_proj linear layer of the self-attention module at the specified layer.
    """
    o_proj_inputs = [] # List to store the captured inputs to o_proj (用于存储捕获到的 o_proj 输入的列表)

    def _hook_o_proj(module, inputs, outputs):
        """
        钩子函数，用于捕获 o_proj 层的输入。
        Hook function to capture the input to the o_proj layer.
        'inputs' is a tuple, and for a linear layer, inputs[0] is the actual input tensor.
        'inputs' 是一个元组，对于线性层，inputs[0] 是实际的输入张量。
        """
        # 获取 O_proj 的输入 (Get the input of O_proj)
        o_proj_inputs.append(inputs[0])  # O_proj 的输入 (Input to O_proj)

    # 假设模型结构类似于 LLaMA 或 GPT，其中解码器层有 self_attn 和 o_proj
    # Assuming a model structure similar to LLaMA or GPT, where decoder layers have self_attn and o_proj
    # Note: model.get_decoder().layers might need adjustment based on the specific model architecture.
    # 注意: model.get_decoder().layers 可能需要根据具体的模型架构进行调整。
    # Common paths: model.model.layers (LLaMA), model.transformer.h (GPT-2), model.decoder.layers (some T5-like)
    # 常见的路径：model.model.layers (LLaMA), model.transformer.h (GPT-2), model.decoder.layers (某些类T5模型)

    # Attempt to access layers, robustly trying common paths
    # 尝试访问层，稳健地尝试常用路径
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'): # LLaMA-like structure
            # 类 LLaMA 结构
            decoder_layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'): # GPT-2 like structure
             # 类 GPT-2 结构
            decoder_layers = model.transformer.h
        elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'): # T5-like structure for decoder
            # 类 T5 解码器结构
            decoder_layers = model.decoder.layers
        elif hasattr(model, 'layers'): # Directly has layers (直接拥有 layers 属性)
            decoder_layers = model.layers
        else:
            raise AttributeError("Could not find the main list of layers in the model. Tried common paths like 'model.layers', 'transformer.h', 'decoder.layers'. Please adapt the path.")
            # 未能在模型中找到主要的层列表。已尝试如 'model.layers', 'transformer.h', 'decoder.layers' 等常见路径。请调整路径。

        self_attn_module = decoder_layers[layer].self_attn # Get the self-attention module for the specified layer (获取指定层的自注意力模块)
        o_proj_layer = self_attn_module.o_proj # Get the o_proj linear layer (获取 o_proj 线性层)
    except AttributeError as e:
        raise AttributeError(f"Error accessing model components: {e}. Please ensure the model structure and layer index are correct. The path used was based on common Transformer architectures.")
        # 访问模型组件时出错: {e}。请确保模型结构和层索引正确。所用路径基于常见的 Transformer 架构。
    except IndexError:
        raise IndexError(f"Layer index {layer} is out of bounds for the model's layers (found {len(decoder_layers)} layers).")
        # 层索引 {layer} 超出了模型层的范围 (找到 {len(decoder_layers)} 层)。


    # 注册 O_proj 层的钩子 (Register the hook on the o_proj layer)
    handle = o_proj_layer.register_forward_hook(_hook_o_proj)

    try:
        # 执行前向传播以触发钩子 (Perform a forward pass to trigger the hook)
        # Tokenize prompts and move to the model's device
        # 对提示进行分词并移至模型的设备
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        device = next(model.parameters()).device # Get device from model parameters (从模型参数中获取设备)
        inputs = {k: v.to(device) for k, v in inputs.items()} # Move all input tensors to the device (将所有输入张量移至设备)
        _ = model(**inputs) # Perform forward pass, output is not needed here (执行前向传播，此处不需要输出)
    finally:
        handle.remove()  # 确保在完成后移除钩子 (Ensure the hook is removed after completion)

    if not o_proj_inputs:
        raise RuntimeError("The hook to capture o_proj input did not run or did not capture any input. Check model execution and hook registration.")
        # 用于捕获 o_proj 输入的钩子未运行或未捕获任何输入。请检查模型执行和钩子注册。
    return o_proj_inputs[0] # Return the captured input (返回捕获到的输入)