#!/usr/bin/env python3
"""
Simulate AF (Attention–FFN) separation with 2 ranks, 2 ubatches, 27 layers.

Communication pattern:
- Rank 0 (FFN):  each layer: recv 2 from ATTN then send 2 to ATTN.
- Rank 1 (ATTN): layer 0: 2 sends to FFN;
                 layers 1–25: recv 2 from FFN then send 2 to FFN;
                 layer 26: 2 recvs from FFN.

Run (2 GPUs on one machine):
  cd /path/to/vllm && torchrun --nproc_per_node=2 examples/pynccl_p2p_cudagraph.py

Or with 2 nodes (1 GPU each), set MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

NUM_UBATCHES = 2
NUM_LAYERS = 27
# Per-ubatch buffer shape (same on both ranks)
SHAPE = (64, 2048)
DTYPE = torch.bfloat16
# FFN rank does recv then send every layer; ATTN rank does 2 sends at layer 0, then recv+send, then 2 recvs at last
FFN_RANK = 0
ATTN_RANK = 1


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "This script expects exactly 2 ranks."

    dist.init_process_group(backend="gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    comm = PyNcclCommunicator(group=_get_default_group(), device=device)
    if comm.disabled:
        print(f"Rank {rank}: PyNcclCommunicator disabled, exit.")
        return

    stream = torch.cuda.current_stream(device)

    # Buffers: each rank has send/recv buffers for both ubatches
    if rank == FFN_RANK:
        # FFN uses one buffer for both ubatches (reused in for loop)
        comm_buf = torch.empty(SHAPE, dtype=DTYPE, device=device)
    else:
        send_ubatch = [
            torch.zeros(SHAPE, dtype=DTYPE, device=device),
            torch.zeros(SHAPE, dtype=DTYPE, device=device),
        ]
        recv_ubatch = [
            torch.empty(SHAPE, dtype=DTYPE, device=device),
            torch.empty(SHAPE, dtype=DTYPE, device=device),
        ]

    def attn_layer_0():
        """ATTN rank: layer 0 — 2 sends to FFN (ubatch0, ubatch1)."""
        comm.send(send_ubatch[0], dst=FFN_RANK, stream=stream)
        comm.send(send_ubatch[1], dst=FFN_RANK, stream=stream)

    def attn_layer_mid(layer_idx):
        """ATTN rank: layers 1..25 — recv 2 from FFN then send 2 to FFN."""
        comm.recv(recv_ubatch[0], src=FFN_RANK, stream=stream)
        comm.recv(recv_ubatch[1], src=FFN_RANK, stream=stream)
        comm.send(send_ubatch[0], dst=FFN_RANK, stream=stream)
        comm.send(send_ubatch[1], dst=FFN_RANK, stream=stream)

    def attn_layer_last():
        """ATTN rank: layer 26 — recv 2 from FFN then send 2 (to match FFN recv=54)."""
        comm.recv(recv_ubatch[0], src=FFN_RANK, stream=stream)
        comm.recv(recv_ubatch[1], src=FFN_RANK, stream=stream)
        comm.send(send_ubatch[0], dst=FFN_RANK, stream=stream)
        comm.send(send_ubatch[1], dst=FFN_RANK, stream=stream)

    def ffn_layer(layer_idx):
        """FFN rank: recv then send per ubatch; layer 26 only recv (no send)."""
        for ub in range(NUM_UBATCHES):
            comm.recv(comm_buf, src=ATTN_RANK, stream=stream)
            if layer_idx < NUM_LAYERS - 1:
                comm.send(comm_buf, dst=ATTN_RANK, stream=stream)

    def run_all_layers():
        """Execute all 27 layers (used for warmup and graph capture)."""
        for layer_idx in range(NUM_LAYERS):
            if rank == FFN_RANK:
                ffn_layer(layer_idx)
            else:
                if layer_idx == 0:
                    attn_layer_0()
                elif layer_idx < NUM_LAYERS - 1:
                    attn_layer_mid(layer_idx)
        if rank == ATTN_RANK:
            attn_layer_last()

    # Warmup (eager run)
    run_all_layers()
    torch.cuda.synchronize(device)
    dist.barrier()
    print(f"Rank {rank}: warmup OK")

    # CUDA graph capture
    capture_stream = torch.cuda.Stream(device=device)
    graph = torch.cuda.CUDAGraph()
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.graph(graph, stream=capture_stream):
        run_all_layers()
    torch.cuda.synchronize(device)
    dist.barrier()
    print(f"Rank {rank}: graph captured")

    # Replay
    num_replays = 4
    for step in range(num_replays):
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        graph.replay()
        torch.cuda.synchronize(device)
        print(f"jcz after replay {step}")
        dist.barrier()
    print(f"Rank {rank}: AF simulation {NUM_LAYERS} layers, {num_replays} replays OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
