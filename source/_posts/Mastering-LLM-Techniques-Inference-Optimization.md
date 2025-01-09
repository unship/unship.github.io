---
title: "Mastering LLM Techniques: Inference Optimization"
date: 2025-01-09 15:59:41
tags:
---

- [Mastering LLM Techniques: Inference Optimization](#mastering-llm-techniques-inference-optimization)
  - [Understanding LLM inference](#understanding-llm-inference)
    - [Prefill phase or processing the input](#prefill-phase-or-processing-the-input)
    - [Decode phase or generating the output](#decode-phase-or-generating-the-output)
    - [Batching](#batching)
    - [Key-value caching](#key-value-caching)
    - [LLM memory requirement](#llm-memory-requirement)
  - [LLM Scaling up LLMs with model parallelization](#llm-scaling-up-llms-with-model-parallelization)
    - [Pipeline parallelism](#pipeline-parallelism)
    - [Tensor parallelism](#tensor-parallelism)
    - [Sequence parallelism](#sequence-parallelism)
  - [Optimizing the attention mechanism](#optimizing-the-attention-mechanism)
    - [Multi-head attention](#multi-head-attention)
    - [Multi-query attention](#multi-query-attention)
    - [Grouped-query attention](#grouped-query-attention)
    - [Flash attention](#flash-attention)
  - [Efficient management of KV cache with paging](#efficient-management-of-kv-cache-with-paging)
  - [Model optimization techniques](#model-optimization-techniques)
    - [Quantization](#quantization)
    - [Sparsity](#sparsity)
    - [Distillation](#distillation)
  - [Model serving techniques](#model-serving-techniques)
    - [In-flight batching](#in-flight-batching)
    - [Speculative inference](#speculative-inference)
  - [Conclusion](#conclusion)

<!---
TODO 增加导读
- transformers
- 存算比
- atlas 硬件架构
-->

带着问题阅读

- 我有一个intervl 8B的模型, atlas 300 v pro能跑么?
- 这个模型性能很差, 怎么知道目前别的硬件上已知的优化, 在这次的目标硬件上已经做了?

翻译自 https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

# Mastering LLM Techniques: Inference Optimization

Stacking transformer layers to create large models results in better accuracies, few-shot learning capabilities, and even near-human emergent abilities on a wide range of language tasks. These foundation models are expensive to train, and they can be memory- and compute-intensive during inference (a recurring cost). The most popular [large language models (LLMs)](https://www.nvidia.com/en-us/glossary/data-science/large-language-models/) today can reach tens to hundreds of billions of parameters in size and, depending on the use case, may require ingesting long inputs (or contexts), which can also add expense. For example, [retrieval-augmented generation](https://developer.nvidia.com/blog/tag/retrieval-augmented-generation-rag/) (RAG) pipelines require putting large amounts of information into the input of the model, greatly increasing the amount of processing work the LLM has to do.

堆叠 transformer 层以创建大型模型可以在多种语言任务上带来更好的准确性、少量样本学习能力，甚至接近人类的涌现能力。这些基础模型训练成本高，并且在推理过程中可能需要大量内存和计算资源（这是一种持续的成本）。目前最流行的大型语言模型（LLM）可以达到数百亿到数百亿参数的规模，并且根据使用场景，可能需要处理较长的输入（或上下文），这也会增加开销。例如，[检索增强生成](https://developer.nvidia.com/blog/tag/retrieval-augmented-generation-rag/)（RAG）管道需要将大量信息输入到模型中，极大增加了 LLM 需要处理的工作量。

<!--
! on a wide range of language tasks. 中文翻译放到前面很好
-->

This post discusses the most pressing challenges in LLM inference, along with some practical solutions. Readers should have a basic understanding of [transformer architecture](https://arxiv.org/pdf/1706.03762.pdf) and the attention mechanism in general. It is essential to have a grasp of the intricacies of LLM inference, which we will address in the next section.

这篇文章讨论了 LLM 推理中最紧迫的挑战，并提供了一些实用的解决方案。读者应该对[transformer 架构](https://arxiv.org/pdf/1706.03762.pdf)和注意力机制有基本了解。理解 LLM 推理的复杂性是至关重要的，我们将在下一节中进行详细讨论。

## Understanding LLM inference

Most of the popular decoder-only LLMs (GPT-3, for example) are pretrained on the causal modeling objective, essentially as next-word predictors. These LLMs take a series of tokens as inputs, and generate subsequent tokens autoregressively until they meet a stopping criteria (a limit on the number of tokens to generate or a list of stop words, for example) or until it generates a special `<end>` token marking the end of generation. This process involves two phases: the prefill phase and the decode phase.

大多数流行的仅使用解码器的 LLM（例如 GPT-3）在因果建模目标上进行预训练，本质上是作为下一个词预测器。这些 LLM 将一系列标记作为输入，并自回归地生成后续标记，直到满足停止条件（例如生成的标记数量限制或停止词列表），或者直到生成一个标志着生成的结束的`<end>`标记。这个过程包括两个阶段：预填充阶段和解码阶段。

Note that _tokens_ are the atomic parts of language that a model processes. One token is approximately four English characters. All inputs in natural language are converted to tokens before inputting into the model.

请注意，_标记_ 是模型处理的语言原子部分。一个标记大约相当于四个英文字母。在将自然语言输入模型之前，所有输入都会被转换为标记。

### Prefill phase or processing the input

In the prefill phase, the LLM processes the input tokens to compute the intermediate states (keys and values), which are used to generate the "first" new token. Each new token depends on all the previous tokens, but because the full extent of the input is known, at a high level this is a matrix-matrix operation that's highly parallelized. It effectively saturates GPU utilization.

在预填充阶段，LLM 处理输入标记以计算中间状态（键和值），这些状态用于生成"第一个"新标记。每个新标记依赖于所有先前的标记，但由于已知整个输入，从高层次来看，这是一个高度并行化的矩阵-矩阵操作，能够有效饱和 GPU 的利用率。

### Decode phase or generating the output

In the decode phase, the LLM generates output tokens autoregressively one at a time, until a stopping criteria is met. Each sequential output token needs to know all the previous iterations' output states (keys and values). This is like a matrix-vector operation that underutilizes the GPU compute ability compared to the prefill phase. The speed at which the data (weights, keys, values, activations) is transferred to the GPU from memory dominates the latency, not how fast the computation actually happens. In other words, this is a memory-bound operation.

在解码阶段，LLM 自回归地一次生成一个输出标记，直到满足停止条件。每个顺序输出标记需要知道所有先前迭代的输出状态（键和值）。这就像一个矩阵-向量操作，与预填充阶段相比，它没有充分利用 GPU 的计算能力。数据（权重、键、值、激活）的传输速度从内存到 GPU 主导了延迟，而不是计算本身的速度。换句话说，这是一个受内存限制的操作。

Many of the inference challenges and corresponding solutions featured in this post concern the optimization of this decode phase: efficient attention modules, managing the keys and values effectively, and others.

本文介绍的许多推理挑战和相应的解决方案都涉及到优化解码阶段：高效的注意力模块、有效管理键和值等。

Different LLMs may use different tokenizers, and thus, comparing output tokens between them may not be straightforward. When comparing inference throughput, even if two LLMs have similar tokens per second output, they may not be equivalent if they use different tokenizers. This is because corresponding tokens may represent a different number of characters.

不同的 LLM 可能使用不同的 tokenizer，因此，比较它们之间的输出标记可能并不直接。即使两个 LLM 的推理吞吐量（每秒标记数）相似，但如果它们使用不同的分词器，它们性能也可能不相同。这是因为对应的标记可能表示不同数量的字符。

### Batching

The simplest way to improve GPU utilization, and effectively throughput, is through batching. Since multiple requests use the same model, the memory cost of the weights is spread out. Larger batches getting transferred to the GPU to be processed all at once will leverage more of the compute available.

提高 GPU 利用率和有效吞吐量的最简单方法是通过批处理。由于多个请求使用相同的模型，权重的内存开销得以分摊。将更大的批次一次性传输到 GPU 进行处理，可以更充分地利用可用的计算资源。

Batch sizes, however, can only be increased up to a certain limit, at which point they may lead to a memory overflow. To better understand why this happens requires looking at key-value (KV) caching and LLM memory requirements.

然而，批次大小只能增加到一定限度，超过这个限度可能会导致内存溢出。要更好地理解为什么会发生这种情况，需要查看键值（KV）缓存和 LLM 的内存需求。

Traditional batching (also called static batching) is suboptimal. This is because for each request in a batch, the LLM may generate a different number of completion tokens, and subsequently they have different execution times. As a result, all requests in the batch must wait until the longest request is finished, which can be exacerbated by a large variance in the generation lengths. There are methods to mitigate this, such as in-flight batching, which will be discussed later.

传统的批处理（也称为静态批处理）是次优的。因为对于批次中的每个请求，LLM 可能生成不同数量的标记，随之而来的是不同的执行时间。因此，所有请求必须等到最长的请求完成，这在生成长度差异较大的情况下会更加严重。有一些方法可以缓解这种问题，比如在处理中的批处理（in-flight batching），稍后将讨论。

### Key-value caching

One common optimization for the decode phase is KV caching. The decode phase generates a single token at each time step, but each token depends on the key and value tensors of all previous tokens (including the input tokens' KV tensors computed at prefill, and any new KV tensors computed until the current time step).

解码阶段的一种常见优化是键值（KV）缓存。在解码阶段，每个时间步只生成一个标记，但每个标记都依赖于所有先前标记的键和值张量（包括预填充阶段计算的输入标记的 KV 张量，以及当前时间步之前计算的新 KV 张量）。

To avoid recomputing all these tensors for all tokens at each time step, it's possible to cache them in GPU memory. Every iteration, when new elements are computed, they are simply added to the running cache to be used in the next iteration. In some implementations, there is one KV cache for each layer of the model.

为了避免在每个时间步为所有标记重新计算这些张量，可以将它们缓存在 GPU 内存中。每次迭代中，当计算出新的元素时，只需将其添加到现有缓存中，以便在下一次迭代中使用。在某些实现中，模型的每一层都有一个独立的 KV 缓存。

> An illustration of KV caching depicted in Prefill and Decode phases. Prefill is a highly parallelized operation where the KV tensors of all input tokens can be computed simultaneously. During decode, new KV tensors and subsequently the output token at each step is computed autoregressively.

> KV 缓存的示意图可以展示预填充阶段和解码阶段的区别。在预填充阶段，这是一个高度并行的操作，所有输入标记的 KV 张量可以同时计算。而在解码阶段，每一步自回归地计算新的 KV 张量，然后生成对应的输出标记。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/key-value-caching_.png)

Figure 1. An illustration of the key-value caching mechanism

图 1.键值缓存机制说明

### LLM memory requirement

In effect, the two main contributors to the GPU LLM memory requirement are model weights and the KV cache.

实际上， LLM 的 GPU 内存需求的两个主要贡献者是模型权重和 KV 缓存。

- **Model weights:** Memory is occupied by the model parameters. As an example, a model with 7 billion parameters (such as [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b/blob/main/params.json)), loaded in 16-bit precision (FP16 or BF16) would take roughly $7B * sizeof(FP16) ~= 14 GB$ in memory.
- **KV caching**: Memory is occupied by the caching of self-attention tensors to avoid redundant computation.
- **模型权重**：内存被模型参数占用。例如，一个具有 70 亿参数的模型（如 [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b/blob/main/params.json)），以 16 位精度（FP16 或 BF16）加载时，大约需要 $7B \times \text{sizeof}(\text{FP16}) \approx 14 \, \text{GB}$ 内存。
- **KV 缓存**：内存被自注意力张量的缓存占用，以避免冗余计算。

With batching, the KV cache of each of the requests in the batch must still be allocated separately, and can have a large memory footprint. The formula below delineates the size of the KV cache, applicable to most common LLM architectures today.

在使用批处理时，批次中每个请求的 KV 缓存仍需单独分配，并且可能占用大量内存。以下公式说明了 KV 缓存的大小，该公式适用于当前大多数常见的 LLM 架构。

$$
\text{Size of KV cache per token in bytes} = 2 * (\text{num\_layers}) * (\text{num\_heads} * \text{dim\_head}) * \text{precision\_in\_bytes}
$$

$$
\text{每个 token 的 KV 缓存大小（字节）} = 2 * ({层数}) * ({注意力头数} * {每个头的维度}) * {每个数据的字节数}
$$

The first factor of 2 accounts for the K and V matrices. Commonly, the value of $({num\_heads} * {dim\_head})$ is the same as the hidden_size (or dimension of the model, $d\_model$) of the transformer. These model attributes are commonly found in model cards or associated config files.

第一个系数 2 表示 K 和 V 矩阵。通常，$(\text{num\_heads} * \text{dim\_head})$的值与 transformer 模型的隐藏层大小（或模型的维度，$d\_model$ ）相同。这些模型属性通常可以在模型卡或相关的配置文件中找到。

This memory size is required for each token in the input sequence, across the batch of inputs. Assuming half-precision, the total size of KV cache is given by the formula below.

这个内存大小是针对输入序列中的每个 token，在整个输入批次中都需要的。假设使用半精度（half-precision），KV 缓存的总大小可以通过以下公式计算：

$$
\text{Total size of KV cache in bytes} = (\text{batch\_size}) * (\text{sequence\_length}) * 2 * (\text{num\_layers}) * (\text{hidden\_size}) * \text{sizeof(FP16)}
$$

$$
\text{KV 缓存的总大小（字节）} = (\text{批次大小}) * (\text{序列长度}) * 2 * (\text{层数}) * (\text{隐藏层大小}) * \text{sizeof(FP16)}
$$

For example, with a Llama 2 7B model in 16-bit precision and a batch size of 1, the size of the KV cache will be $1 * 4096 * 2 * 32 * 4096 * 2 bytes$, which is ~2 GB.

例如，对于一个 16 位精度的 Llama 2 7B 模型，批次大小为 1 时，KV 缓存的大小将为：$1 * 4096 * 2 * 32 * 4096 * 2 \, \text{字节}$，约为 2GB。

Managing this KV cache efficiently is a challenging endeavor. Growing linearly with batch size and sequence length, the memory requirement can quickly scale. Consequently, it limits the throughput that can be served, and poses challenges for long-context inputs. This is the motivation behind several optimizations featured in this post.

有效管理 KV 缓存是一个具有挑战性的任务。由于内存需求随着批次大小和序列长度线性增长，缓存的大小可以迅速扩大。因此，它限制了可处理的吞吐量，并对长上下文输入提出了挑战。这也是本文中介绍的几种优化方法的动机所在。

<!---
TODO 缓存大小是线性增长么?
-->

## LLM Scaling up LLMs with model parallelization

One way to reduce the per-device memory footprint of the model weights is to distribute the model over several GPUs. Spreading the memory and compute footprint enables running larger models, or larger batches of inputs. Model parallelization is a necessity to train or infer on a model requiring more memory than available on a single device, and to make training times and inference measures (latency or throughput) suitable for certain use cases. There are several ways of parallelizing the model based on how the model weights are split.

减少每个设备上模型权重内存占用的一种方法是将模型分布到多个 GPU 上。通过分散内存和计算负载，可以运行更大的模型或更大的输入批次。模型并行化是训练或推理需要比单个设备可用内存更多的模型时的必要手段，并且可以使训练时间和推理指标（如延迟或吞吐量）适应某些用例。根据模型权重的拆分方式，有几种并行化模型的方法。

Note that data parallelism is also a technique often mentioned in the same context as the others listed below. In this, weights of the model are copied over multiple devices, and the (global) batch size of inputs is sharded across each of the devices into microbatches. It reduces the overall execution time by processing larger batches. However, it is a training time optimization that is less relevant during inference.

请注意，数据并行性也是与下述其他技术常常一起提到的一种方法。在数据并行性中，模型的权重会复制到多个设备上，输入的（全局）批次大小会被切分成微批次，分配到每个设备上。通过处理更大的批次，它可以减少整体执行时间。然而，它是一个训练时间的优化，在推理过程中相关性较小。

### Pipeline parallelism

Pipeline parallelism involves sharding the model (vertically) into chunks, where each chunk comprises a subset of layers that is executed on a separate device. Figure 2a is an illustration of four-way pipeline parallelism, where the model is sequentially partitioned and a quarter subset of all layers are executed on each device. The outputs of a group of operations on one device are passed to the next, which continues executing the subsequent chunk. F_n and B_n indicate forward and backward passes respectively on device n. The memory requirement for storing model weights on each device is effectively quartered.

流水线并行将模型垂直分片为多个块，每个块由一部分层组成，并在单独的设备上执行。图 2a 展示了四路流水线并行的示例，其中模型被顺序划分，每个设备执行所有层的四分之一子集。一组操作的输出从一个设备传递到下一个设备，后者继续执行后续的块。<span dir="">F_n</span> 和 <span dir="">B_n</span> 分别表示设备 n 上的前向传播和反向传播。存储模型权重的内存需求在每个设备上有效地减少到四分之一。

The main limitation of this method is that, due to the sequential nature of the processing, some devices or layers may remain idle while waiting for the output (activations, gradients) of previous layers. This results in inefficiencies or "pipeline bubbles" in both the forward and backward passes. In Figure 2b, the white empty areas are the large pipeline bubbles with naive pipeline parallelism where devices are idle and underutilized.

这种方法的主要限制是，由于处理的顺序性，在等待前一层的输出（激活、梯度）时，某些设备或层可能会处于空闲状态。这导致了效率低下或在前向和反向传播中出现所谓的"流水线气泡"。在图 2b 中，白色空白区域表示简单流水线并行中较大的流水线气泡，此时设备处于空闲且利用率较低。

Microbatching can mitigate this to some extent, as shown in Figure 2c. The global batch size of inputs is split into sub-batches, which are processed one by one, with gradients being accumulated at the end. Note that F\_{n,m} and B\_{n,m} indicate forward and backward passes respectively on device n with microbatch m. This approach shrinks the size of pipeline bubbles, but it does not completely eliminate them.

微批处理（Microbatching）在一定程度上可以缓解这一问题，如图 2c 所示。输入的全局批量大小被划分为多个子批量，一个接一个地处理，并在最后累积梯度。注意，F\_{n,m} 和 B\_{n,m} 分别表示设备 n 在微批 m 上的前向传播和反向传播。这种方法缩小了流水线气泡的大小，但不能完全消除它们。

> Depiction of four-way pipeline parallelism. (a) Model is partitioned across layers in 4 parts, each subset executed on a separate device. (b) Naive pipeline parallelism results in large pipeline bubbles and GPU under-utilization. (c) Micro-batching reduces the size of pipeline bubbles, and improves GPU utilization.

> 四路流水线并行示意图。a 模型按层划分为 4 部分，每个子集在单独的设备上执行。b 简单流水线并行导致大的流水线气泡和 GPU 利用率低。c 微批处理减少了流水线气泡的大小，提高了 GPU 的利用率。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/four-way-pipeline-parallelism.png)

Figure 2. An illustration of four-way pipeline parallelism. Credit: [_GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism_](https://arxiv.org/pdf/1811.06965.pdf)
图 2. 四路流水线并行示例图。来源: [_GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism_](https://arxiv.org/pdf/1811.06965.pdf)

### Tensor parallelism

Tensor parallelism involves sharding (horizontally) individual layers of the model into smaller, independent blocks of computation that can be executed on different devices. Attention blocks and multi-layer perceptron (MLP) layers are major components of transformers that can take advantage of tensor parallelism. In multi-head attention blocks, each head or group of heads can be assigned to a different device so they can be computed independently and in parallel.

张量并行是将模型的每一层水平分片为较小的、独立的计算块，这些块可以在不同设备上执行。注意力块和多层感知机（MLP）层是 transformers 中可以利用张量并行的主要组件。在多头注意力块中，每个头或头组都可以分配到不同的设备上，以便独立并行计算。

> Illustration of Tensor Parallelism in MLPs and Self-Attention Layers. In MLPs, the weight matrix is partitioned across multiple devices, enabling simultaneous computation on a batch of inputs using the split weights. In self-attention layers, the multiple attention heads are naturally parallel and can be distributed across devices.>
> MLP 和自注意力层中张量并行的示意图。在 MLP 中，权重矩阵分布到多个设备上，使用分片权重可以同时处理一批输入。在自注意力层中，多个注意力头天然是并行的，可以分配到不同设备上。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/tensor-parallelsim-mlp-self-attention-layers_.png)

Figure 3. Illustration of tensor parallelism in multi-layer perceptron (MLP) and self-attention layers. Credit: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)

图 3. 多层感知机（MLP）和自注意力层中的张量并行示意图。来源: [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/pdf/1909.08053.pdf)

Figure 3a shows an example of two-way tensor parallelism on a two-layer MLP, with each layer represented by a rounded box. Within the first layer, the weight matrix $A is split into $A_1$ and $A_2$. The computations $XA_1$ and $XA_2$ can be independently executed on the same batch (f is an identity operation) of inputs X on two different devices. This effectively halves the memory requirement of storing weights on each device. A reduction operation g combines the outputs in the second layer.

图 3a 展示了一个两路张量并行的两层 MLP 示例，每一层用一个圆角矩形表示。在第一层中，权重矩阵 $A$ 被分为 $A_1$ 和 $A_2$。计算 $XA_1$ 和 $XA_2$ 可以在相同输入批量 $X$（f 是身份操作）上独立地在两个不同设备上执行。这有效地将每个设备存储权重的内存需求减少了一半。一个归约操作 g 在第二层中组合输出。

Figure 3b is an example of two-way tensor parallelism in the self-attention layer. The multiple attention heads are parallel by nature and can be split across devices.

图 3b 是自注意力层中两路张量并行的示例。多个注意力头本质上是并行的，可以分配到不同设备上。

### Sequence parallelism

Tensor parallelism has limitations, as it requires layers to be divided into independent, manageable blocks. It's not applicable to operations like LayerNorm and Dropout, which are instead replicated across the tensor-parallel group. While LayerNorm and Dropout are computationally inexpensive, they do require a considerable amount of memory to store (redundant) activations.

张量并行存在一定的局限性，因为它需要将层划分为独立且易于管理的块。对于像 LayerNorm 和 Dropout 这样的操作，张量并行无法适用，这些操作通常在张量并行组中被复制。虽然 LayerNorm 和 Dropout 的计算开销较小，但它们需要大量内存来存储（冗余的）激活。

As shown in [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf), these operations are independent across the input sequence, and these ops can be partitioned along that "sequence-dimension," making them more memory efficient. This is called sequence parallelism.

正如 [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf) 所示，这些操作在输入序列上是独立的，可以沿着"序列维度"划分，从而提高内存效率。这被称为序列并行。

> Illustration of a transformer layer with both Tensor parallelism and Sequence parallelism. Sequence parallelism is applicable for operations like LayerNorm and Dropout, which are not well-suited for tensor parallelism.
> 具有张量并行和序列并行的变换器层示意图。序列并行适用于 LayerNorm 和 Dropout 等不适合张量并行的操作。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/transformer-layer-tensor-and-sequence-parallelism.png)

Figure 4. An illustration of a transformer layer with both tensor and sequence parallelism. Credit: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf)

图 4. 同时具有张量并行和序列并行的变换器层示意图。来源: [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/pdf/2205.05198.pdf)

Techniques for model parallelism are not exclusive and can be used in conjunction. They can help scale and reduce the per-GPU memory footprint of LLMs, but there are also optimization techniques specifically for the attention module.

模型并行的技术并非互斥，它们可以结合使用。这些方法有助于扩展大型语言模型（LLMs）并减少每个 GPU 的内存占用。此外，还有一些针对注意力模块的专门优化技术。

## Optimizing the attention mechanism

The scaled dot-product attention (SDPA) operation maps query and key-value pairs to an output, as described in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf). 缩放点积注意力（Scaled Dot-Product Attention，SDPA）操作将查询（Query）和键值对（Key-Value pairs）映射到输出，如 [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) 中所述。

### Multi-head attention

As an enhancement to the SDPA, executing the attention layer multiple times in parallel with different, learned projections of the Q, K, and V matrices, enables the model to jointly attend to information from different representational subspaces at different positions. These subspaces are learned independently, providing the model with a richer understanding of different positions in the input.

作为对 SDPA 的增强，注意力层可以通过对 Q、K 和 V 矩阵的不同学习投影进行多次并行执行，从而使模型能够联合关注输入中不同位置的不同表示子空间。这些子空间是独立学习的，赋予模型对输入不同位置的更丰富理解。

As depicted in Figure 5, the outputs from the multiple parallel attention operations are concatenated and linearly projected to combine them. Each parallel attention layer is called a 'head,' and this approach is called multi-head attention (MHA).

如图 5 所示，多个并行注意力操作的输出被连接起来，并通过线性投影组合在一起。每个并行注意力层称为一个"头"（Head），这种方法称为多头注意力（MHA）。

In the original work, each attention head operates on a reduced dimension of the model (such as $d\_{model}/8$ when using eight parallel attention heads. This keeps the computational cost similar to single-head attention.

在原始研究中，每个注意力头操作的维度被减少（例如，当使用八个并行注意力头时，单个头的维度为 $d\_{model}/8$）。这使得其计算成本与单头注意力相似。

> An illustration of the scaled dot-product attention and multi-head attention.
> 缩放点积注意力和多头注意力的示意图。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/scaled-dot-product-attention-and-multi-head-attention.png)

Figure 5. An illustration of the scaled dot-product attention (left) and multi-head attention (right), which is simply multiple SDPA heads in parallel. Credit: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

图 5. 缩放点积注意力（左）和多头注意力（右）的示意图，多头注意力即多个并行的 SDPA 头。来源: [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

### Multi-query attention

One of the inference optimizations to MHA, called multi-query attention (MQA), as proposed in [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150), shares the keys and values among the multiple attention heads. The query vector is still projected multiple times, as before.

多查询注意力（Multi-Query Attention，MQA）是对多头注意力（MHA）的一种推理优化，由 [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150) 提出。它通过在多个注意力头中共享键（Key）和值（Value）来实现优化，但查询向量（Query）仍像之前一样被多次投影。

While the amount of computation done in MQA is identical to MHA, the amount of data (keys, values) read from memory is a fraction of before. When bound by memory-bandwidth, this enables better compute utilization. It also reduces the size of the KV-cache in memory, allowing space for larger batch sizes.

尽管 MQA 的计算量与 MHA 相同，但从内存中读取的数据量（键和值）却大大减少。在受内存带宽限制的情况下，这可以提高计算利用率。同时，它也减少了 KV 缓存的内存占用，从而为更大的批量大小腾出空间。

The reduction in key-value heads comes with a potential accuracy drop. Additionally, models that need to leverage this optimization at inference need to train (or [at least fine-tuned](https://arxiv.org/pdf/2305.13245.pdf) with ~5% of training volume) with MQA enabled.

然而，减少键值头的数量可能会导致精度下降。此外，需要在推理阶段利用这种优化的模型，必须在训练（或 [至少微调](https://arxiv.org/pdf/2305.13245.pdf)，约占训练量的 5%）时启用 MQA。

### Grouped-query attention

[Grouped-query attention](https://arxiv.org/pdf/2305.13245v2.pdf) (GQA) strikes a balance between MHA and MQA by projecting key and values to a few groups of query heads (Figure 6). Within each of the groups, it behaves like multi-query attention.

[分组查询注意力](https://arxiv.org/pdf/2305.13245v2.pdf)（Grouped-Query Attention，GQA）在 MHA 和 MQA 之间取得平衡，通过将键和值投影到少量的查询头组中实现优化（图 6）。在每个组内，其行为类似于多查询注意力。

Figure 6 shows that multi-head attention has multiple key-value heads (left). Grouped-query attention (center) has more key-value heads than one, but fewer than the number of query heads, which is a balance between memory requirement and model quality. Multi-query attention (right) has a single key-value head to help save memory.

图 6 显示了多头注意力有多个键值头（左）。分组查询注意力（中）相比 MQA 有更多的键值头，但少于查询头的数量，在内存需求和模型质量之间取得了平衡。多查询注意力（右）只有一个键值头，从而节省内存。

> Different attention mechanisms compared. Left: Multi-head attention has multiple key-value heads. Right: Multi-query attention has a single key-value head, which reduces memory requirements. Center: Grouped-query attention has a few key-value heads, balancing memory and model quality.
> 不同注意力机制的比较。左：多头注意力有多个键值头。右：多查询注意力只有一个键值头，减少了内存需求。中：分组查询注意力有少量键值头，在内存需求和模型质量之间取得平衡。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/comparison-attention-mechanisms.png)

Figure 6. A comparison of different attention mechanisms. Credit: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245v2.pdf)

图 6. 不同注意力机制的比较。来源: [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245v2.pdf)

Models originally trained with MHA, can be "uptrained" with GQA using a fraction of the original training compute. They attain quality close to MHA while maintaining a computational efficiency closer to MQA. [Llama 2 70B](https://ai.meta.com/llama/) is an example of a model that leverages GQA.

最初用 MHA 训练的模型可以通过较少的训练计算量"再训练"到 GQA。它们在保持接近 MHA 的质量的同时，计算效率接近 MQA。[Llama 2 70B](https://ai.meta.com/llama/) 是一个利用 GQA 的模型示例。

Optimizations like MQA and GQA help reduce the memory required by KV caches by reducing the number of key and value heads that are stored. There may still be inefficiencies in how this KV cache is managed. Of a different flavor than optimizing the attention module itself, the next section presents a technique for more efficient KV cache management.

像 MQA 和 GQA 这样的优化通过减少存储的键和值头的数量，降低了 KV 缓存所需的内存。然而，KV 缓存的管理仍可能存在效率问题。与直接优化注意力模块不同，下一节将介绍一种更高效管理 KV 缓存的技术。

### Flash attention

<!--
TODO ratio of compute and storage, why this optimization work
-->

Another way of optimizing the attention mechanism is to modify the ordering of certain computations to take better advantage of the memory hierarchy of GPUs. Neural networks are generally described in terms of layers, and most implementations are laid out that way as well, with one kind of computation done on the input data at a time in sequence. This doesn't always lead to optimal performance, since it can be beneficial to do more calculations on values that have already been brought into the higher, more performant levels of the memory hierarchy.

优化注意力机制的另一种方法是调整某些计算的顺序，从而更好地利用 GPU 的内存层次结构。神经网络通常按层次描述，大多数实现也按照这种方式进行，每次对输入数据执行一种计算操作。但这种顺序不一定能实现最佳性能，因为对已经加载到更高性能内存层次中的数据进行更多计算可能更高效。

Fusing multiple layers together during the actual computation can enable minimizing the number of times the GPU needs to read from and write to its memory and to group together calculations that require the same data, even if they are parts of different layers in the neural network.

在实际计算中融合多个层次的操作可以减少 GPU 需要读取和写入内存的次数，同时将需要相同数据的计算组合在一起，即使它们属于神经网络中不同的层次。

One very popular fusion is FlashAttention, an I/O aware exact attention algorithm, as detailed in [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). _Exact attention_ means that it is mathematically identical to the standard multi-head attention (with variants available for multi-query and grouped-query attention), and so can be swapped into an existing model architecture or even an already-trained model with no modifications.

一种非常流行的融合方法是 FlashAttention，这是一种 I/O 感知的精确注意力算法，详见 [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)。**精确注意力**意味着它在数学上与标准的多头注意力完全相同（也有适用于多查询和分组查询注意力的变体），因此可以直接插入现有模型架构或已经训练好的模型中，无需任何修改。

_I/O aware_ means it takes into account some of the memory movement costs previously discussed when fusing operations together. In particular, FlashAttention uses "tiling" to fully compute and write out a small part of the final matrix at once, rather than doing part of the computation on the whole matrix in steps, writing out the intermediate values in between.

_I/O 感知_ 意味着它在融合操作时考虑了内存移动成本。尤其是，FlashAttention 使用"分块"（tiling）技术，一次完全计算并写出最终矩阵的一小部分，而不是分步骤对整个矩阵进行部分计算并在中间写出中间值。

Figure 7 shows the tiled FlashAttention computation pattern and the memory hierarchy on a 40 GB GPU. The chart on the right shows the relative speedup that comes from fusing and reordering the different components of the Attention mechanism.

图 7 展示了分块 FlashAttention 的计算模式以及在 40 GB GPU 上的内存层次结构。右侧的图表显示了通过融合和重新排序注意力机制不同组件所带来的相对加速效果。

> Diagram depicting the memory hierarchy and the FlashAttention computation.
> 显示内存层次结构和 FlashAttention 计算模式的示意图。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/flash-attention-computation-pattern-memory-hierarchy-gpu.png)

Figure 7. The tiled FlashAttention computation pattern and the memory hierarchy on a 40 GB GPU. Credit: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

图 7. 分块 FlashAttention 的计算模式和 40 GB GPU 上的内存层次结构。来源: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)

## Efficient management of KV cache with paging

At times, KV caches are statically "over-provisioned" to account for the largest possible input (the supported sequence length) because the size of inputs is unpredictable. For example, if the supported maximum sequence length of a model is 2,048, then regardless of the size of input and the generated output in a request, a reservation of size 2,048 would be made in memory. This space may be contiguously allocated, and often, much of it remains unused, leading to memory waste or fragmentation. This reserved space is tied up for the lifetime of the request.

KV 缓存有时会被静态"过度预留"以适应可能的最大输入（支持的序列长度），因为输入大小难以预测。例如，如果模型支持的最大序列长度是 2,048，那么无论请求中的输入或生成的输出大小如何，都会在内存中预留一个大小为 2,048 的空间。这些空间可能被连续分配，但通常大部分空间未被使用，导致内存浪费或碎片化。这些预留空间会在请求的整个生命周期内被占用。

> An illustration of memory wastage and fragmentation due to over-provisioning and inefficient management of KV cache.
> 由于过度预留和 KV 缓存管理效率低下导致的内存浪费和碎片化示意图。

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/memory-wastage-fragmentation-inefficient-kv-cache.png)

Figure 8. An illustration of memory wastage and fragmentation due to over-provisioning and inefficient KV cache management. Credit: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180.pdf) 图 8. 由于过度预留和 KV 缓存管理效率低下导致的内存浪费和碎片化示意图。来源: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180.pdf)

Inspired by paging in operating systems, the [PagedAttention](https://vllm.ai/) algorithm enables storing continuous keys and values in noncontiguous space in memory. It partitions the KV cache of each request into blocks representing a fixed number of tokens, which can be stored non-contiguously.

受到操作系统分页机制的启发，[PagedAttention](https://vllm.ai/) 算法使得可以在内存中以非连续方式存储连续的键和值。它将每个请求的 KV 缓存分成表示固定数量标记的块，这些块可以以非连续的方式存储。

These blocks are fetched as required during attention computation using a block table that keeps account. As new tokens are generated, new block allocations are made. The size of these blocks is fixed, eliminating inefficiencies arising from challenges like different requests requiring different allocations. This significantly limits memory wastage, enabling larger batch sizes (and, consequently, throughput).

在注意力计算过程中，这些块会根据需要通过一个块表（block table）进行提取和管理。当生成新的标记时，会分配新的块。这些块的大小是固定的，从而消除了由于不同请求需要不同分配而带来的低效问题。这显著减少了内存浪费，并支持更大的批量大小（进而提高吞吐量）。

## Model optimization techniques

So far, we've discussed the different ways LLMs consume memory, some of the ways memory can be distributed across several different GPUs, and optimizing the attention mechanism and KV cache. There are also several model optimization techniques to reduce the memory use on each GPU by making modifications to the model weights themselves. GPUs also have dedicated hardware for accelerating operations on these modified values, providing even more speedups for models.

到目前为止，我们已经讨论了大型语言模型（LLMs）如何消耗内存、如何将内存分布在多个 GPU 上，以及优化注意力机制和 KV 缓存的方法。此外，还有一些模型优化技术，可以通过修改模型权重本身来减少每个 GPU 的内存使用。GPU 还配备了专用硬件，用于加速对这些修改后的权重值的操作，从而为模型提供更大的加速效果。

### Quantization

_Quantization_ is the process of reducing the precision of a model's weights and activations. Most models are trained with 32 or 16 bits of precision, where each parameter and activation element takes up 32 or 16 bits of memory--a single-precision floating point. However, most deep learning models can be effectively represented with eight or even fewer bits per value.

_量化_ 是指减少模型权重和激活值精度的过程。大多数模型使用 32 位或 16 位精度进行训练，其中每个参数和激活元素占用 32 或 16 位内存（单精度浮点）。然而，大多数深度学习模型可以用每个值 8 位甚至更低的精度有效地表示。

Figure 9 shows the distribution of values before and after one possible method of quantization. In this case, some precision is lost to rounding, and some dynamic range is lost to clipping, allowing the values to be represented in a much smaller format.

图 9 显示了量化前后值分布的变化。在这种情况下，通过四舍五入会丢失一些精度，而通过裁剪会丢失一些动态范围，从而使值可以用更小的格式表示。

> Two distribution plots, one showing the full range of values at high precision and another showing the compressed and rounded range at low precision.
> 两个分布图，一个显示高精度值的完整范围，另一个显示低精度下的压缩和四舍五入范围。
> ![ ](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/quantization-value-distribution.png)

Figure 9. The distribution of values before and after one possible method of quantization 图 9. 量化前后值分布的变化 Reducing the precision of a model can yield several benefits. If the model takes up less space in memory, you can fit larger models on the same amount of hardware. Quantization also means you can transfer more parameters over the same amount of bandwidth, which can help to accelerate models that are bandwidth-limited.

减少模型精度有以下几个好处：

1. 如果模型占用的内存更少，可以在相同的硬件上容纳更大的模型。
2. 量化还可以使更多参数通过相同带宽传输，从而加速带宽受限的模型。

There are many different quantization techniques for LLMs involving reduced precision on either the activations, the weights, or both. It's much more straightforward to quantize the weights because they are fixed after training. However, this can leave some performance on the table because the activations remain at higher precisions. GPUs don't have dedicated hardware for multiplying INT8 and FP16 numbers, so the weights must be converted back into a higher precision for the actual operations.

在 LLM 中，有多种量化技术涉及降低激活值、权重或两者的精度。量化权重相对简单，因为训练后权重是固定的。然而，仅量化权重可能会错失一些性能提升，因为激活值仍然保留较高精度。此外，GPU 缺乏专用硬件来直接计算 INT8 和 FP16 的乘法运算，因此权重必须在实际操作时转换回更高精度。

It's also possible to quantize the activations, the inputs of transformer blocks and network layers, but this comes with its own challenges. Activation vectors often contain outliers, effectively increasing their dynamic range and making it more challenging to represent these values at a lower precision than with the weights.

激活值（transformer 块和网络层的输入）的量化也是可能的，但存在一些挑战。激活向量中通常包含离群值，这会显著增加其动态范围，因而比权重更难以用低精度表示。

One option is to find out where those outliers are likely to show up by passing a representative dataset through the model, and choosing to represent certain activations at a higher precision than others (LLM.int8()). Another option is to borrow the dynamic range of the weights, which are easy to quantize, and reuse that range in the activations.

以下是两种解决方法：

1. **预测离群值的位置**：通过让模型处理一个具有代表性的数据集，找到激活值中可能出现离群值的位置，然后选择在某些激活值上使用较高精度（如 LLM.int8() 方法）。
2. **借用权重的动态范围**：权重易于量化，可以复用其动态范围来表示激活值。

### Sparsity

Similar to quantization, it's been shown that many deep learning models are robust to pruning, or replacing certain values that are close to 0 with 0 itself. _Sparse matrices_ are matrices where many of the elements are 0. These can be expressed in a condensed form that takes up less space than a full, dense matrix. 与量化类似，已有研究表明，许多深度学习模型对于剪枝是鲁棒的，或者说可以用 0 替代接近 0 的某些值。_稀疏矩阵_ 是指矩阵中许多元素为 0 的矩阵。这些矩阵可以以一种压缩的形式表示，占用的空间比完整的密集矩阵要小。

> A sparse matrix represented in a compressed format.
> 以压缩格式表示的稀疏矩阵。
> ![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/sparse-matrix-compressed-format_.png)

Figure 10. A sparse matrix represented in a compressed format consisting of non-zero data values and their corresponding two-bit indices

图 10. 以压缩格式表示的稀疏矩阵，包括非零数据值及其对应的两位索引

GPUs in particular have hardware acceleration for a certain kind of _structured sparsity_, where two out of every four values are represented by zeros. Sparse representations can also be combined with quantization to achieve even greater speedups in execution. Finding the best way to represent large language models in a sparse format is still an active area of research, and offers a promising direction for future improvements to inference speeds.

特别是，GPU 对某种类型的 _结构化稀疏性_ 具有硬件加速支持，其中每四个值中的两个由零表示。稀疏表示还可以与量化结合，进一步加速执行速度。如何以稀疏格式表示大型语言模型的最佳方法仍是一个活跃的研究领域，并为未来推理速度的提升提供了一个有前景的方向。

### Distillation

Another approach to shrinking the size of a model is to transfer its knowledge to a smaller model through a process called _distillation_. This process involves training a smaller model (called a student) to mimic the behavior of a larger model (a teacher).

另一种缩小模型规模的方法是通过一个叫做 _蒸馏_ 的过程，将知识转移到一个更小的模型中。这个过程涉及训练一个较小的模型（称为学生模型）来模仿一个较大模型（称为教师模型）的行为。

Successful examples of distilled models include [DistilBERT](https://arxiv.org/abs/1910.01108), which compresses a BERT model by 40% while retaining 97% of its language understanding capabilities at a speed 60% faster.

成功的蒸馏模型实例包括 [DistilBERT](https://arxiv.org/abs/1910.01108)，它在压缩 BERT 模型 40% 的同时，保持了 97% 的语言理解能力，并且速度提高了 60%。

While distillation in LLMs is an active field of research, the general approach was first described for neural networks in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531):

尽管在大语言模型（LLM）中的蒸馏仍是一个活跃的研究领域，但这种一般方法最早是在 [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) 中描述的：

- The student network is trained to mirror the performance of a larger teacher network, using a loss function that measures the discrepancy between their outputs. This objective is in addition to potentially including the original loss function of matching the student's outputs with the ground-truth labels.
- The teacher's outputs that are matched can be the very last layer (called _logits_) or intermediate layer activations.
- 学生网络通过使用一个衡量其输出与教师网络输出差异的损失函数来训练，以此来模拟教师网络的表现。这个目标是在可能的情况下，除了匹配学生输出与真实标签的原始损失函数之外的额外目标。
- 与教师输出匹配的部分可以是最后一层（称为 _logits_）或中间层激活。

Figure 11 shows a general framework for knowledge distillation. The logits of the teacher are soft targets that the student optimizes for using a distillation loss. Other distillation methods may use other measures of loss to "distill" knowledge from the teacher.

图 11 展示了一个知识蒸馏的通用框架。教师的 logits 是软目标，学生通过使用蒸馏损失来优化它们。其他蒸馏方法可能使用不同的损失度量来从教师中"提炼"知识。

> Figure depicting a general framework for knowledge distillation using a distillation loss between the logits of the teacher and student.
> 展示使用蒸馏损失在教师和学生的 logits 之间进行知识蒸馏的通用框架。

![ ](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/knowledge-distillation-general-framework.png)

Figure 11. A general framework for knowledge distillation. Credit: [Knowledge Distillation: A Survey](https://arxiv.org/pdf/2006.05525.pdf) 图 11. 知识蒸馏的通用框架。来源：[Knowledge Distillation: A Survey](https://arxiv.org/pdf/2006.05525.pdf)

An alternative approach to distillation is to use data synthesized by the teacher for supervised training of a student LLM, which is especially useful when human annotations are scarce or not available. [Distilling Step by Step!](https://arxiv.org/abs/2305.02301) goes one step further by extracting rationales from a teacher LLM in addition to the labels that serve as ground truth. These rationales serve as intermediate reasoning steps to train smaller student LLMs in a data-efficient way.

蒸馏的另一种方法是使用教师合成的数据来监督训练学生 LLM，这在人工标注稀缺或无法获取时特别有用。[Distilling Step by Step!](https://arxiv.org/abs/2305.02301) 更进一步，通过从教师 LLM 中提取推理过程和作为真实标签的标签，帮助训练更小的学生 LLM。这些推理过程作为中间推理步骤，以数据高效的方式训练学生 LLM。

It's important to note that many state-of-the-art LLMs today have restrictive licenses that prohibit using their outputs to train other LLMs, making it challenging to find a suitable teacher model.

需要注意的是，许多当前最先进的 LLM 都有限制性许可，禁止使用它们的输出训练其他 LLM，这使得找到合适的教师模型变得具有挑战性。

## Model serving techniques

Model execution is frequently memory-bandwidth bound--in particular, bandwidth-bound in the weights. Even after applying all the model optimizations previously described, it's still very likely to be memory bound. So you want to do as much as possible with your model weights when they are loaded. In other words, try doing things in parallel. Two approaches can be taken:

模型执行通常受限于内存带宽，特别是在权重部分。即使应用了之前描述的所有模型优化，仍然很可能会受到内存带宽的限制。因此，在加载模型权重时，尽可能多地利用它们是很重要的。换句话说，尽量尝试并行处理。可以采取两种方法：

- **In-flight batching** involves executing multiple different requests at the same time.
- **Speculative inference** involves executing multiple different steps of the sequence in parallel to try to save time.
- **运行时批处理** 通过同时执行多个不同的请求来进行处理。
- **推测推理** 通过并行执行序列中的多个不同步骤来节省时间。

### In-flight batching

LLMs have some unique execution characteristics that can make it difficult to effectively batch requests in practice. A single model can be used simultaneously for a variety of tasks that look very different from one another. From a simple question-and-answer response in a chatbot to the summarization of a document or the generation of a long chunk of code, workloads are highly dynamic, with outputs varying in size by several orders of magnitude.

大语言模型（LLM）具有一些独特的执行特性，这可能使得在实践中有效地批处理请求变得困难。一个模型可以同时用于多种看起来非常不同的任务。从聊天机器人的简单问答响应到文档的摘要，或者生成一大段代码，工作负载高度动态，输出大小差异可能跨越几个数量级。

This versatility can make it challenging to batch requests and execute them in parallel effectively--a common optimization for serving neural networks. This could result in some requests finishing much earlier than others.

这种多功能性使得批处理请求并有效地并行执行变得具有挑战性----这是为神经网络服务时常见的优化方法。这可能导致一些请求比其他请求提前完成许多。

To manage these dynamic loads, many LLM serving solutions include an optimized scheduling technique called continuous or in-flight batching. This takes advantage of the fact that the overall text generation process for an LLM can be broken down into multiple iterations of execution on the model.

为了管理这些动态负载，许多 LLM 服务解决方案包括一种优化的调度技术，称为连续或飞行中批处理。这利用了大语言模型的整体文本生成过程可以分解为多个执行迭代的事实。

With in-flight batching, rather than waiting for the whole batch to finish before moving on to the next set of requests, the server runtime immediately evicts finished sequences from the batch. It then begins executing new requests while other requests are still in flight. In-flight batching can therefore greatly increase the overall GPU utilization in real-world use cases.

通过飞行中批处理，服务器运行时不会等到整个批处理完成后再处理下一批请求，而是立即将已完成的序列从批处理中驱逐出去。它随后开始执行新的请求，同时其他请求仍在进行中。因此，飞行中批处理可以大大提高现实场景中的 GPU 利用率。

### Speculative inference

Also known as speculative sampling, assisted generation, or blockwise parallel decoding, speculative inference is a different way of parallelizing the execution of LLMs. Normally, GPT-style large language models are autoregressive models that generate text token by token.

推测推理，也被称为推测采样、辅助生成或块级并行解码，是一种并行化执行大语言模型（LLM）的方法。通常，GPT 风格的大型语言模型是自回归模型，逐个生成文本的 token。

Every token that is generated relies on all of the tokens that come before it to provide context. This means that in regular execution, it's impossible to generate multiple tokens from the same sequence in parallel--you have to wait for the nth token to be generated before you can generate n+1.

每个生成的 token 都依赖于它之前的所有 token 以提供上下文。这意味着在常规执行中，无法并行生成同一序列中的多个 token----你必须等到第 n 个 token 生成后，才能生成 n+1。

Figure 12 shows an example of speculative inference in which a draft model temporarily predicts multiple future steps that are verified or rejected in parallel. In this case, the first two predicted tokens in the draft are accepted, while the last is rejected and removed before continuing with the generation.

图 12 显示了推测推理的一个例子，其中草稿模型临时预测多个未来步骤，这些步骤在并行中被验证或拒绝。在这种情况下，草稿中前两个预测的 token 被接受，而最后一个被拒绝并在继续生成之前移除。

> From the prompt
> 从提示中

![](https://developer-blogs.nvidia.com/wp-content/uploads/2023/11/speculative-inference-example_.png)

Figure 12. An example of speculative inference. Credit: [Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115)
图 12. 推测推理的一个例子。来源：[Blockwise Parallel Decoding for Deep Autoregressive Models](https://arxiv.org/abs/1811.03115)

Speculative sampling offers a workaround. The basic idea of this approach is to use some "cheaper" process to generate a draft continuation that is several tokens long. Then, execute the main "verification" model at multiple steps in parallel, using the cheap draft as "speculative" context for the execution steps where it is needed.

推测采样提供了一个解决方法。这种方法的基本思路是使用一些"较便宜"的过程生成一个草稿延续，长度为多个 token。然后，在多个步骤中并行执行主要的"验证"模型，使用便宜的草稿作为执行步骤所需的"推测"上下文。

If the verification model generates the same tokens as the draft, then you know to accept those tokens for the output. Otherwise, you can throw out everything after the first non-matching token, and repeat the process with a new draft.

如果验证模型生成的 token 与草稿相同，则接受这些 token 作为输出。否则，你可以丢弃第一个与草稿不匹配的 token 之后的所有内容，并使用新的草稿重复该过程。

There are many different options for how to generate draft tokens, and each comes with different tradeoffs. You can train multiple models, or fine-tune multiple heads on a single pretrained model, that predict tokens that are multiple steps in the future. Or, you can use a small model as the draft model, and a larger, more capable model as the verifier.

生成草稿 token 的方法有很多种，每种都有不同的权衡。你可以训练多个模型，或者在一个预训练模型上微调多个头部，预测未来多个步骤的 token。或者，你可以使用一个小模型作为草稿模型，而将一个更大、更强大的模型作为验证器。

## Conclusion

This post outlines many of the most popular solutions to help optimize and serve LLMs efficiently, be it in the data center or at the edge on a PC. Many of these techniques are optimized and available through [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0), an open-source library consisting of the TensorRT deep learning compiler alongside optimized kernels, preprocessing and postprocessing steps, and multi-GPU/multi-node communication primitives for groundbreaking performance on NVIDIA GPUs. To learn more, see [Optimizing Inference on Large Language Models with NVIDIA TensorRT-LLM, Now Publicly Available](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/).

本文概述了帮助优化和高效服务大语言模型（LLM）的许多流行解决方案，无论是在数据中心还是在 PC 边缘。许多这些技术通过 [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0) 进行了优化并提供了开源支持，该库包含 TensorRT 深度学习编译器以及优化的内核、预处理和后处理步骤，以及支持多 GPU/多节点通信的原语，以在 NVIDIA GPU 上提供突破性的性能。欲了解更多信息，请参见 [Optimizing Inference on Large Language Models with NVIDIA TensorRT-LLM, Now Publicly Available](https://developer.nvidia.com/blog/optimizing-inference-on-llms-with-tensorrt-llm-now-publicly-available/)。

NVIDIA TensorRT-LLM is now supported by [NVIDIA Triton Inference Server](https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/), enabling enterprises to serve multiple AI models concurrently across different AI frameworks, hardware accelerators, and deployment models with peak throughput and minimum latency.

NVIDIA TensorRT-LLM 现在由 [NVIDIA Triton Inference Server](https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/) 支持，使企业能够在不同的 AI 框架、硬件加速器和部署模型中并行服务多个 AI 模型，提供最高吞吐量和最低延迟。
