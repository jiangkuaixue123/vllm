# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AFD metadata definitions for communication between attention and
FFN workers."""

import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from abc import ABC, abstractmethod

#TODO(yxj):move to AFDExtraFields
from vllm_ascend.ascend_forward_context import MoECommType
from dataclasses import dataclass, field
from typing import Dict


class AFDRecvHandle(ABC):
    """
    Abstract base class for AFD receive handles.
    
    This provides a handle interface for managing asynchronous AFD operations,
    allowing waiting for completion of data transfer operations.
    """
    @abstractmethod
    def __init__(self, handle: Any):
        """Initialize the AFD receive handle.
        
        Args:
            handle: Backend-specific handle object
        """
        raise NotImplementedError

    @abstractmethod
    def wait(self):
        """Wait for the operation associated with this handle to complete.
        
        Blocks until the data transfer or computation is finished.
        """
        raise NotImplementedError


class FFNNeedForwardData:
    def __init__(self,
                 moe_comm_type:Optional[MoECommType] = None,
                 num_input_tokens:int = 0,
                 with_prefill:bool = False,
                 total_num_scheduled_tokens:int = 0,
                 is_dummy_run:bool = False):
        self.moe_comm_type = moe_comm_type
        self.num_input_tokens = num_input_tokens
        self.with_prefill = with_prefill
        self.total_num_scheduled_tokens = total_num_scheduled_tokens


@dataclass
class M2NAFDConnectorMetadata:
    def __init__(self):
        self.topk_idx = None
        self.topk_weights = None
        self.moe_expert_num = 0
        self.scale = None
        self.handle = None
        self.quant_mode = 0
        self.aiv_num = 0
        self.batch_size = 0
        self.h = 0
        self.k = 0
        self.expert_token_nums_type = 0
        self.expand_x_type = torch.float16
        
@dataclass
class CAMM2NAFDConnectorMetadata:
    def __init__(self, moe_expert_num=0,
        shared_expert_num = 0, scale=None, handle=None, quant_mode=0,
        aiv_num=0, batch_size=0, h=0, k=0):
        self.moe_expert_num = moe_expert_num
        self.shared_expert_num = shared_expert_num
        self.scale = scale
        self.handle = handle
        self.quant_mode = quant_mode
        self.aiv_num = aiv_num
        self.batch_size = batch_size
        self.h = h
        self.k = k

@dataclass
class CAMP2PAFDConnectorMetadata:
    def __init__(self, moe_expert_num=0,
        shared_expert_num = 0, scale=None, handle=None, quant_mode=0,
        aiv_num=0, batch_size=0, h=0, k=0):
        self.moe_expert_num = moe_expert_num
        self.shared_expert_num = shared_expert_num
        self.scale = scale
        self.handle = handle
        self.quant_mode = quant_mode
        self.aiv_num = aiv_num
        self.batch_size = batch_size
        self.h = h
        self.k = k

@dataclass
class AFDExtraFields:
    """Additional field specifically for storing AFDconnectors"""
    custom_fields: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs):
        self.custom_fields.update(kwargs)

@dataclass
class AFDConnectorMetadata:
    """Lightweight AFD metadata containing core information needed for
    communication."""
    layer_idx: int
    stage_idx: int
    seq_lens: list[
        int]  # Length of each sequence, supports variable length and
    # multiple sequences
    dtype: torch.dtype
    device: torch.device
    topk_idx: Optional[torch.Tensor] = None # indices token which expert to be sended
    topk_weights: Optional[torch.Tensor] = None # the expert weights
    moe_expert_num: Optional[int] = None # number of moe experts
    shared_expert_num: Optional[int] = None # number of share experts
    scale: Optional[torch.Tensor] = None #  quant scale
    expertTokenNumsOut: Optional[torch.Tensor] = None # The number of tokens received by each expert is used as input for the subsequent GMM.
    send_handle_list: Optional[list[Any]] = None # the communication handles (list of Work objects returned by torch.distributed.isend)
    recv_handle_list: Optional[list[Any]] = None # the communication handles (list of Work objects returned by torch.distributed.irecv)

    # TODO(jcz): need fix vllm_ascend dependency
    ffn_need_forward_data: Optional[FFNNeedForwardData] = None
    m2n_afdconnector_data: Optional[M2NAFDConnectorMetadata] = None
    cam_m2n_afdconnector_data: Optional[CAMM2NAFDConnectorMetadata] = None
    cam_p2p_afdconnector_data: Optional[CAMP2PAFDConnectorMetadata] = None
    topk_weights: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None
    row_idx: Optional[torch.Tensor] = None
    
    # Optional fields for debugging and extensibility
    request_id: Optional[str] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        """Validate data consistency."""
        if not self.seq_lens:
            raise ValueError("seq_lens cannot be empty")
        if any(length <= 0 for length in self.seq_lens):
            raise ValueError("All sequence lengths must be positive")

    @property
    def total_tokens(self) -> int:
        """Total number of tokens."""
        return sum(self.seq_lens)

    @property
    def num_sequences(self) -> int:
        """Number of sequences."""
        return len(self.seq_lens)

    @property
    def is_single_sequence(self) -> bool:
        """Whether this is a single sequence (attention side characteristic)."""
        return len(self.seq_lens) == 1

    @property
    def is_multi_sequence(self) -> bool:
        """Whether this is multiple sequences (FFN side characteristic)."""
        return len(self.seq_lens) > 1

    @classmethod
    def create_attention_metadata(
            cls,
            layer_idx: int,
            stage_idx: int,
            seq_len: int,
            dtype: torch.dtype,
            device: torch.device,
            request_id: Optional[str] = None,
            ffn_need_forward_data:Optional[FFNNeedForwardData] = None,
            m2n_afdconnector_data:Optional[M2NAFDConnectorMetadata] = None,
            cam_m2n_afdconnector_data:Optional[CAMM2NAFDConnectorMetadata] = None,
            cam_p2p_afdconnector_data:Optional[CAMP2PAFDConnectorMetadata] = None,
            # extra_fields: AFDExtraFields = field(default_factory=AFDExtraFields),
            topk_weights: Optional[torch.Tensor] = None,
            topk_ids: Optional[torch.Tensor] = None,
            row_idx: Optional[torch.Tensor] = None) -> "AFDConnectorMetadata":
        """Create metadata for attention side (single sequence)."""
        return cls(layer_idx=layer_idx,
                   stage_idx=stage_idx,
                   seq_lens=[seq_len],
                   dtype=dtype,
                   device=device,
                   request_id=request_id,
                #    timestamp=time.time(),
                   ffn_need_forward_data=ffn_need_forward_data,
                   m2n_afdconnector_data=m2n_afdconnector_data,
                   cam_m2n_afdconnector_data=cam_m2n_afdconnector_data,
                   cam_p2p_afdconnector_data=cam_p2p_afdconnector_data,
                   topk_weights=topk_weights,
                   topk_ids=topk_ids,
                   row_idx=row_idx,
                #    extra_fields = extra_fields
                   )

    @classmethod
    def create_ffn_metadata(
            cls,
            layer_idx: int,
            stage_idx: int,
            seq_lens: list[int],
            dtype: torch.dtype,
            device: torch.device,
            request_id: Optional[str] = None) -> "AFDConnectorMetadata":
        """Create metadata for FFN side (multiple sequences)."""
        return cls(
            layer_idx=layer_idx,
            stage_idx=stage_idx,
            seq_lens=seq_lens.copy(),  # Prevent external modification
            dtype=dtype,
            device=device,
            request_id=request_id,
            timestamp=time.time())

    def get_split_indices(self) -> list[int]:
        """Get tensor split indices for FFN side output splitting."""
        if len(self.seq_lens) <= 1:
            return []

        indices = []
        cumsum = 0
        for length in self.seq_lens[:-1]:  # Exclude the last one
            cumsum += length
            indices.append(cumsum)
        return indices

    def validate_tensor_shape(self, tensor_shape: tuple[int, ...]) -> bool:
        """Validate if tensor shape is consistent with metadata."""
        if len(tensor_shape) < 1:
            return False
        return tensor_shape[0] == self.total_tokens

    def to_dict(self) -> dict:
        """Convert to dictionary format for serialization and debugging."""
        return {
            "layer_idx": self.layer_idx,
            "stage_idx": self.stage_idx,
            "seq_lens": self.seq_lens,
            "dtype": self.dtype,
            "device": self.device,
            "total_tokens": self.total_tokens,
            "num_sequences": self.num_sequences,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        """Friendly string representation."""
        return (f"AFDConnectorMetadata(layer={self.layer_idx}, "
                f"stage={self.stage_idx}, seq_lens={self.seq_lens}, "
                f"total_tokens={self.total_tokens}, dtype={self.dtype}, "
                f"device={self.device}, request_id={self.request_id})")
