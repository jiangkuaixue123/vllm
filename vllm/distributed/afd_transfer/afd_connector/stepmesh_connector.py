# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import re
import subprocess
from collections import deque
import numpy as np
from typing import Optional
import fserver_lib as ps
import torch
import threading
import torch.cuda.nvtx as nvtx
import torch.distributed.distributed_c10d as c10d

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    TensorMetadata,
    init_afd_process_group,
    init_model_parallel_group,
)
from vllm.logger import init_logger
from vllm.forward_context import (
    DPMetadata,
    get_forward_context,
)

from .base import AFDConnectorBase
from .metadata import AFDConnectorMetadata

logger = init_logger(__name__)

import torch._dynamo
torch._dynamo.config.ignore_logger_methods.add(logger.info)
# torch._dynamo.config.ignore_logger_methods = True

lib = torch.library.Library("ps", "DEF")


lib.define("seq_add_one(Tensor(a!) seq) -> ()")
lib.define("write_flag(Tensor(a!) flag, Tensor seq) -> ()")
lib.define("wait_flag(Tensor(a!) flag, Tensor seq) -> ()")
lib.define("wait_and_recv(Tensor(a!) flag, Tensor seq, Tensor(b!) recv_buf) -> ()")



@torch.library.impl(lib, "seq_add_one", "Meta")
def seq_add_one_meta(seq):
    return

@torch.library.impl(lib, "write_flag", "Meta")
def write_flag_meta(flag, seq):
    return

@torch.library.impl(lib, "wait_flag", "Meta")
def wait_flag_meta(flag, seq):
    return

@torch.library.impl(lib, "wait_and_recv", "Meta")
def wait_and_recv_meta(flag, seq, recv_buf):
    return


@torch.library.impl(lib, "seq_add_one", "CUDA")
def seq_add_one_cuda(seq):
    ps.seq_add_one(seq)

@torch.library.impl(lib, "write_flag", "CUDA")
def write_flag_cuda(flag, seq):
    ps.write_flag(flag, seq)

@torch.library.impl(lib, "wait_flag", "CUDA")
def wait_flag_cuda(flag, seq):
    ps.wait_flag(flag, seq)

@torch.library.impl(lib, "wait_and_recv", "CUDA")
def wait_and_recv_cuda(flag, seq, recv_buf):
    ps.wait_flag(flag, seq)

DEBUG = os.environ.get("STEPMESH_CONNECTOR_DEBUG", "false").lower() == "true"


class StepMeshAFDConnector(AFDConnectorBase):
    """StepMesh-based implementation of AFD connector.

    This connector uses StepMesh parameter server for communication between
    attention workers and FFN servers.
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):
        """Initialize StepMesh AFD connector.

        Args:
            rank: Global rank of this process
            local_rank: Local rank within the node
            config: VllmConfig containing AFD configuration
        """
        self.afd_config = config.afd_config
        self.rank = rank
        self.local_rank = local_rank
        self.server_rank = self.afd_config.afd_server_rank
        self.num_recv_times = (
            self.afd_config.num_ffn_servers
            if self.afd_config.afd_role == "attention"
            else self.afd_config.num_attention_servers
        )
        parallel_config = config.parallel_config
        self.world_size = (
            parallel_config.tensor_parallel_size
            * parallel_config.pipeline_parallel_size
            * parallel_config.data_parallel_size
        )
        if getattr(config.model_config.hf_config, "text_config", None) is not None:
            self.num_hidden_layers: int = (
                config.model_config.hf_config.text_config.num_hidden_layers
            )
        else:
            self.num_hidden_layers: int = (
                config.model_config.hf_config.num_hidden_layers
            )
        self._initialized = False
        self.signal = None
        self.num_stages = 1
        self.recv_counter = 0

        # Metadata tracking for new interface
        self._current_comm_handles = None
        self._current_metadata = None

        logger.info(f"{rank=}, {local_rank=}, {self.afd_config=}, {self.world_size=}")

        if self.afd_config.afd_role == "attention":
            self.events: deque = deque(maxlen=self.num_stages)
            self.max_num_tokens = config.scheduler_config.max_num_batched_tokens
            self.comm_stream = torch.cuda.Stream()
            self.recv_buffer: list[list[torch.Tensor]] = [
                [
                    torch.empty(
                        (
                            self.max_num_tokens,
                            config.model_config.hf_config.hidden_size,
                        ),
                        dtype=torch.bfloat16,
                        device=torch.device("cuda"),
                    ).contiguous()
                    for _ in range(self.num_recv_times)
                ]
                for _ in range(self.num_stages)
            ]
            self.send_buffer: list[torch.Tensor] = [
                torch.empty(
                    (self.max_num_tokens, config.model_config.hf_config.hidden_size),
                    dtype=torch.bfloat16,
                    device=torch.device("cuda"),
                ).contiguous()
                for _ in range(self.num_stages)
            ]

            self.signal_flag_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)
            self.ack_flag_host = torch.zeros(1, dtype=torch.int64, pin_memory=True)

            self.signal_flag_dev = ps.map_pinned_tensor(
                self.signal_flag_host, int(local_rank)
            )
            self.ack_flag_dev = ps.map_pinned_tensor(
                self.ack_flag_host, int(local_rank)
            )

            self.sequence_tensor = torch.zeros(
                1, dtype=torch.int64, device=f"cuda:{local_rank}"
            )
            self.thread_stop = threading.Event()

            expected_sequence = 1

            def cpu_handle_thread():
                nonlocal expected_sequence
                stage_id = 0
                node_rank_offset = int(self.rank * 1e6)

                while True:
                    signal_value = self.signal_flag_host.item()
                    if signal_value < expected_sequence:
                        if self.thread_stop.is_set():
                            break
                        continue
                    seq_len = 1024
                    recv_buff = [t[:seq_len] for t in self.recv_buffer[stage_id]]
                    recv_key = [stage_id + 1000]

                    send_buff = [self.send_buffer[stage_id][:seq_len]]
                    send_key = [stage_id + node_rank_offset]
                    logger.info(f"Attn-{self.local_rank}: cpu handle thread, push pull start, {signal_value=}")
                    
                    # Use NVTX to trace this operation in the background thread
                    # nvtx.range_push("ps.push_pull_thread")
                    with torch.cuda.stream(self.comm_stream):
                        handler = ps.push_pull(
                            send_buff,
                            send_key,
                            recv_buff,
                            recv_key,
                            need_event=False,
                        )
                    # nvtx.range_pop()
                    
                    logger.info(f"Attn-{self.local_rank}: cpu handle thread, push pull done")
                    
                    # nvtx.range_push("ps.wait_thread")
                    ps.wait(handler, timeout_ms=60_000)
                    # nvtx.range_pop()
                    
                    expected_sequence += 1
                    self.ack_flag_host.fill_(signal_value)
                    logger.info(f"Attn-{self.local_rank}: cpu handle thread, update {self.ack_flag_host=}")

            self.send_attn_output_thread = threading.Thread(
                target=cpu_handle_thread, name="send_attn_output_thread"
            )
            self.send_events = [torch.cuda.Event() for _ in range(27)]
            logger.info(f"Attn-{self.local_rank}: send attn output thread start")
            self.send_attn_output_thread.start()
        else:
            self.max_num_tokens = (
                config.scheduler_config.max_num_batched_tokens
            ) * self.num_recv_times
            self.ret_buffer = torch.empty(
                [self.max_num_tokens, config.model_config.hf_config.hidden_size],
                dtype=torch.bfloat16,
                device=torch.device("cuda"),
            ).contiguous()
            self.recv_attn_output_counter = 0

        logger.info(
            f"{self.afd_config.afd_role}: {self.max_num_tokens=}, {self.num_recv_times=}"
        )

        # StepMesh environment setup
        self._setup_stepmesh_env()

        # Start scheduler subprocess for FFN role (using subprocess to avoid daemon process limitations)
        if (
            self.afd_config.afd_role == "ffn"
            and self.afd_config.afd_server_rank == 0
            and self.local_rank == 0
        ):
            self._start_scheduler_process()

    def close(self) -> None:
        """Close the StepMesh connector and release resources."""
        if self._initialized:
            try:
                ps.finalize()
                self._initialized = False
                logger.info("StepMesh connector closed successfully")
            except Exception as e:
                logger.error(f"Failed to close StepMesh connector: {e}")

        # Clean up scheduler subprocess if it exists
        if hasattr(self, "scheduler_process") and self.scheduler_process is not None:
            try:
                if self.scheduler_process.poll() is None:  # Process is still running
                    logger.info("Terminating scheduler subprocess")
                    self.scheduler_process.terminate()
                    self.scheduler_process.wait(timeout=5)
                    logger.info("Scheduler subprocess terminated successfully")
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Scheduler subprocess did not terminate gracefully, killing it"
                )
                self.scheduler_process.kill()
            except Exception as e:
                logger.error(f"Failed to terminate scheduler subprocess: {e}")

        if (
            hasattr(self, "send_attn_output_thread")
            and self.send_attn_output_thread is not None
        ):
            self.thread_stop.set()
            self.send_attn_output_thread.join()

    def init_afd_connector(self) -> None:
        """Initialize StepMesh connector."""
        if self._initialized:
            return
        try:
            logger.info(f"+++++Start init ps. {self.rank}")
            ps.init()
            logger.info(f"----Finish init ps. {self.rank}")

            # self.signal = ps.SimpleNotify()
            # self.signal.init() # type: ignore

            self._initialized = True
            logger.info(
                f"StepMesh connector initialized successfully as {os.environ.get('DMLC_ROLE')}"
            )

        except ImportError as e:
            raise ImportError(
                f"StepMesh is not available. Please install StepMesh to use "
                f"StepMesh AFD connector. Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize StepMesh connector: {e}") from e

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # -------------------------------------------------------------------------
    #                                attn -> ffn
    # -------------------------------------------------------------------------

    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> None:
        """Send attention output to FFN servers via StepMesh push_pull.

        Args:
            hidden_states: Attention output tensor
            metadata: AFD metadata containing layer_idx, stage_idx, seq_len info

        Returns:
            Any: Event handle for tracking this request
        """
        if not self._initialized:
            raise RuntimeError("StepMesh connector not initialized")

        # Validate metadata consistency
        if not metadata.validate_tensor_shape(hidden_states.shape):
            raise ValueError(
                f"Tensor shape {hidden_states.shape} doesn't match metadata {metadata}"
            )

        if not metadata.is_single_sequence:
            raise ValueError("Attention side should have single sequence")

        self.send_attn_seq_len = metadata.seq_lens[0]
        # self.send_buffer[0][:self.send_attn_seq_len].copy_(hidden_states[:self.send_attn_seq_len], non_blocking=True)
        self.send_buffer[0][:self.send_attn_seq_len].copy_(hidden_states[:self.send_attn_seq_len])

        self.layer_idx = metadata.layer_idx
        torch.ops.ps.seq_add_one(self.sequence_tensor)
        torch.ops.ps.write_flag(self.signal_flag_dev, self.sequence_tensor)
        
    def recv_ffn_output(
        self,
        timeout_ms: Optional[float] = None,
    ) -> torch.Tensor:
        """Wait for FFN computation result from FFN servers.

        Args:
            handle: Event handle returned by send_attn_output

        Returns:
            torch.Tensor: FFN computation result
        """
        if not self._initialized:
            raise RuntimeError("StepMesh connector not initialized")
        
        # try:
        #     torch.ops.ps.wait_flag(self.ack_flag_dev, self.sequence_tensor)
        #     logger.info(f"Attn-{self.local_rank}: wait done {self.layer_idx=}")
        #     stage_idx = 0
        #     seq_len = self.send_attn_seq_len
        #     return self.recv_buffer[stage_idx][0][:seq_len]
        # except Exception as e:
        #     logger.error(f"Failed to wait for FFN output: {e}")
        #     raise RuntimeError(f"StepMesh recv_ffn_output failed: {e}") from e
        
        try:
            # Explicitly pass recv_buffer to the op to establish data dependency for torch.compile
            stage_idx = 0
            recv_buf = self.recv_buffer[stage_idx][0]
            # torch.ops.ps.wait_and_recv(self.ack_flag_dev, self.sequence_tensor, recv_buf)
            torch.ops.ps.wait_flag(self.ack_flag_dev, self.sequence_tensor)

            logger.info(f"Attn-{self.local_rank}: wait done {self.layer_idx=}")
            seq_len = self.send_attn_seq_len
            return recv_buf[:seq_len].clone()
        except Exception as e:
            logger.error(f"Failed to wait for FFN output: {e}")
            raise RuntimeError(f"StepMesh recv_ffn_output failed: {e}") from e

    # -------------------------------------------------------------------------
    #                                ffn -> attn
    # -------------------------------------------------------------------------

    def send_ffn_output(
        self,
        ffn_output: torch.Tensor,
        metadata: AFDConnectorMetadata,
    ) -> None:
        """Send FFN computation result back to attention workers.

        Args:
            ffn_output: Computed FFN output
            metadata: AFD metadata containing seq_lens for splitting and routing info
        """
        logger.info(f"FFN-{self.local_rank}: send ffn output , {ffn_output.shape=}, {metadata=}")
        if not self._initialized:
            raise RuntimeError("StepMesh connector not initialized")

        self.ret_buffer[: ffn_output.shape[0]].copy_(ffn_output)

        try:
            # Use metadata.seq_lens for splitting
            split_indices = metadata.get_split_indices()
            if split_indices:
                split_outputs = torch.split(ffn_output, metadata.seq_lens, dim=0)
            else:
                split_outputs = [ffn_output]

            comm_handles = self._current_comm_handles
            with torch.profiler.record_function("ps.respond_vec"):
                ps.respond_vec(self.ret_buffer, split_outputs, comm_handles)
            self.recv_attn_output_counter += 1

        except Exception as e:
            logger.error(f"Failed to send FFN output: {e}")
            raise RuntimeError(f"StepMesh send_ffn_output failed: {e}") from e

    def recv_attn_output(
        self,
        timeout_ms: Optional[float] = None,
    ) -> tuple[torch.Tensor, AFDConnectorMetadata]:
        """Receive attention output from attention workers (FFN server side).

        Args:
            timeout_ms: Optional timeout in milliseconds

        Returns:
            tuple: (hidden_states, metadata) received from attention workers
        """
        if not self._initialized:
            raise RuntimeError("StepMesh connector not initialized")

        try:
            # batches = self.signal.get_batch() # type: ignore
            logger.info(f"FFN-{self.local_rank}: get batch called")
            with torch.profiler.record_function("ps.get_batch"):
                batches = ps.get_batch()  # type: ignore
            logger.info(f"FFN-{self.local_rank}: get batch finished")

            # Extract tensors and build metadata
            recv_tensors = []
            seq_lens = []
            comm_handles = []

            for node_rank in range(self.num_recv_times):
                tensor = batches[node_rank][1][0]
                comm_id = batches[node_rank][0]

                recv_tensors.append(tensor)
                seq_lens.append(tensor.shape[0])
                comm_handles.append(comm_id)

            # Merge tensors
            merged_tensor = torch.cat(recv_tensors, dim=0)

            self.recv_attn_output_counter = (
                self.recv_attn_output_counter
            ) % self.num_hidden_layers
            stage_idx = 0
            layer_idx = self.recv_attn_output_counter
            logger.info(
                f"FFN-{self.local_rank}: recv_attn_output {layer_idx=}, {stage_idx=}, {self.recv_attn_output_counter=}"
            )
            inferred_metadata = AFDConnectorMetadata.create_ffn_metadata(
                layer_idx=layer_idx,
                stage_idx=stage_idx,
                seq_lens=seq_lens,
                dtype=merged_tensor.dtype,
                device=merged_tensor.device,
                request_id=f"ffn_batch_{stage_idx}_{layer_idx}",
            )

            # Store handles for response
            self._current_comm_handles = comm_handles  # type: ignore
            self._current_metadata = inferred_metadata  # type: ignore

            return merged_tensor, inferred_metadata

        except Exception as e:
            logger.error(f"Failed to receive attention output: {e}")
            raise RuntimeError(f"StepMesh recv_attn_output failed: {e}") from e

    def _setup_stepmesh_env(self) -> None:
        """Setup StepMesh environment variables."""
        # Basic StepMesh configuration based on draft.diff
        if self.afd_config.afd_role == "attention":
            os.environ["DMLC_ROLE"] = "worker"
        elif self.afd_config.afd_role == "ffn":
            os.environ["DMLC_ROLE"] = "server"
        else:
            raise ValueError(f"Invalid AFD role: {self.afd_config.afd_role}")

        os.environ["DMLC_NUM_WORKER"] = str(self.afd_config.num_attention_servers)
        os.environ["DMLC_NUM_SERVER"] = str(self.afd_config.num_ffn_servers)

        os.environ["DMLC_ENABLE_RDMA"] = "ibverbs"
        os.environ["DMLC_INTERFACE"] = "auto"
        os.environ["STEPMESH_SPLIT_QP_LAG"] = os.environ.get(
            "STEPMESH_SPLIT_QP_LAG", "0"
        )  # set 1 for bond
        os.environ["STEPMESH_BIND_CPU_CORE"] = "1"

        os.environ["STEPMESH_GPU"] = os.environ.get(
            "STEPMESH_GPU", str(self.local_rank)
        )

        os.environ["DMLC_PS_ROOT_PORT"] = str(self.afd_config.afd_port)
        os.environ["DMLC_PS_ROOT_URI"] = self.afd_config.afd_host
        os.environ["DMLC_NODE_HOST"] = str(self.afd_config.afd_host)
        os.environ["SCHEDULER_IP"] = str(self.afd_config.afd_host)

        os.environ["DMLC_NODE_RANK"] = str(self.afd_config.afd_server_rank)
        os.environ["DMLC_GROUP_SIZE"] = str(self.world_size)

        os.environ["PS_VERBOSE"] = os.environ.get("PS_VERBOSE", "2")

        logger.info(
            f"StepMesh environment setup: role={os.environ.get('DMLC_ROLE')}, "
            f"num_worker={os.environ.get('DMLC_NUM_WORKER')}, "
            f"num_server={os.environ.get('DMLC_NUM_SERVER')}, "
            f"port={os.environ.get('DMLC_PS_ROOT_PORT')}, "
            f"host={os.environ.get('DMLC_PS_ROOT_URI')}, "
            f"node_rank={os.environ.get('DMLC_NODE_RANK')}, "
            f"gpu={os.environ.get('STEPMESH_GPU')}, "
            f"group_size={os.environ.get('DMLC_GROUP_SIZE')}"
        )

    def _start_scheduler_process(self) -> None:
        """Start scheduler process for FFN role.

        This method launches a separate subprocess to run the StepMesh scheduler
        when the current process is in FFN role.
        """
        try:
            logger.info(
                f"{self.local_rank=}, Starting scheduler subprocess for FFN role, may hang here"
            )
            # Use subprocess.Popen to start scheduler as a separate process
            self.scheduler_process = subprocess.Popen(
                [
                    "python",
                    "-c",
                    "import torch; import fserver_lib as ps; import os; "
                    'os.environ["DMLC_ROLE"] = "scheduler"; '
                    'os.environ["DMLC_INTERFACE"] = "bond0"; '
                    "ps.init(); ps.stop()",
                ],
                env=os.environ.copy(),
            )
            logger.info(
                f"Scheduler subprocess started with PID: {self.scheduler_process.pid}"
            )
        except Exception as e:
            logger.error(f"Failed to start scheduler subprocess: {e}")
            raise RuntimeError(f"Failed to start scheduler subprocess: {e}") from e
