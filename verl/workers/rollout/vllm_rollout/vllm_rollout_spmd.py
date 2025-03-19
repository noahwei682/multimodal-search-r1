# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import time
import numpy as np
from typing import List
from contextlib import contextmanager
from copy import deepcopy
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, get_final_eos_mask, pad_2d_list_to_length, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
import requests
from tqdm import tqdm
import pickle,datetime,uuid
from tools.image_search import call_image_search

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def pad_to_max_stack(tensor_list: List[torch.Tensor], pad_token_id: int, dim: int) -> torch.Tensor:
    assert all([t.ndim==1 for t in tensor_list])
    max_len=max([t.size(0) for t in tensor_list])
    padded_tensor_list=[]
    for t in tensor_list:
        padded_tensor_list.append(torch.cat([t,torch.tensor([pad_token_id]*(max_len-t.size(0)),device=t.device,dtype=t.dtype)],dim=0))
    return torch.stack(padded_tensor_list,dim=dim)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        
        # Add tokenizer
        self.tokenizer = tokenizer

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (B', max_prompt_length), [pad_token_id, pad_token_id, pad_token_id, ..., 127, 63, 9021]
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']  # (B', max_prompt_length), [0, 0, 0, ..., 1, 1, 1]
        position_ids = prompts.batch['position_ids']  # (B', max_prompt_length), [0, 0, 0, ..., 1, 2, 3, ...]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id'] # [151645, 151643] # |im_end|, |end_of_text|

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = [] # length=B', list of dict, 'raw_prompt_ids', 'multi_modal_data'
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'), non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{'prompt_token_ids': raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False
            )

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

        response = [] # list of tuple, B'*R, response_ids without pad_token_id
        for output in outputs:
            for sample_id in range(len(output.outputs)):
                response.append(output.outputs[sample_id].token_ids)

        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)

        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
            if 'multi_modal_inputs' in non_tensor_batch.keys():
                non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'], self.config.n)
            if 'image_urls' in non_tensor_batch.keys():
                non_tensor_batch['image_urls'] = _repeat_interleave(non_tensor_batch['image_urls'], self.config.n)
        
        seq = torch.cat([idx, response], dim=-1) # the whole seq, (B'*R, max_prompt_length+max_response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


class vLLMRollout_MultiTurn_ImSearch(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        # add tokenizer
        self.tokenizer = tokenizer

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        # All for 1st USER prompt
        idx = prompts.batch['input_ids']  # (B'*R, max_prompt_length), left padding with |end_of_text|
        batch_size = idx.size(0)  # B'
        # for logit_log_prob & loss computation
        attention_mask = prompts.batch['attention_mask']  # (B'*R, max_prompt_length), left padding 0
        position_ids = prompts.batch['position_ids']  # (B'*R, max_prompt_length), left padding 0
        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']  # [151645, 151643] -> ｜im_end｜, |end_of_text|
        input_prompt_generation_mask = torch.zeros_like(idx, dtype=attention_mask.dtype, device=attention_mask.device) # (B'*R, max_prompt_length), all 0

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        
        n = 1 if prompts.meta_info.get('validate', False) else self.config.n

        ##### Initialization #####
        vllm_inputs = [] # B*R, list of dict, into -> vllm.engine, each dict with keys: 'prompt_token_ids', 'multi_modal_data', the values are 'raw_prompt_ids' and [PIL.Image]
        multi_turn_response_mask = [] # B*R, list of list of Tensor, for distinguish 'USER tokens' & 'ASSISTANT tokens'
        prefix_prompt_lengths = [] # B*R, list of int, record first round prompt of all trajs

        # We manually repeart trajs for rollout, since some trajs need multi-round self.inference_engine.generate() with `sampling_n=1`
        if 'multi_modal_data' in non_tensor_batch:
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'), non_tensor_batch.pop('multi_modal_data')):
                prefix_length = len(raw_prompt_ids)
                for _ in range(n):
                    # NOTE: use deepcopy to seperate variables
                    vllm_inputs.append(
                        {'prompt_token_ids': deepcopy(raw_prompt_ids), 'multi_modal_data': deepcopy(multi_modal_data)} # raw_prompt_ids: list
                    )
                    multi_turn_response_mask.append(
                        [torch.zeros(prefix_length, dtype=attention_mask.dtype, device=attention_mask.device)] # USER, Mark as 0
                    ) # [torch.Tensor(prefix_length,)]
                    prefix_prompt_lengths.append(
                        prefix_length
                    )
        # We need 'image_urls' for search, the shape should be aligned with B*R
        if 'image_urls' in non_tensor_batch.keys() and not prompts.meta_info.get('validate', False):
            non_tensor_batch['image_urls'] = _repeat_interleave(non_tensor_batch['image_urls'], self.config.n)
        
        ##### Loop Setting #####
        to_generate = list(range(batch_size*n))  # B*R, all trajs' index
        worker_trajs_count = len(to_generate)
        # Add pbar for better monitoring
        with tqdm(total=worker_trajs_count, desc="Worker Rollout Progress", unit="task") as pbar:
            current_iteration = 0
            max_iterations = self.config.max_gen_round
            while current_iteration < max_iterations and len(to_generate) > 0: 
                # Prepare prompts to generation
                idx_to_gen = [] # list of vllm_inputs, at first the length is B'*R
                for i in to_generate:
                    idx_to_gen.append(vllm_inputs[i])

                print(f"[Round #{current_iteration} Rollout START] For THIS round, We hava {len(idx_to_gen)} trajs to complete ...")

                # users can customize different sampling_params at different run
                with self.update_sampling_params(n=1):  # TODO: for validate, do_sample=False
                    outputs = self.inference_engine.generate(
                        prompts=idx_to_gen,  # list of dict
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )

                response = [] # list of tuple, B'*R, valid(no-pad) response_ids with unequal length
                for output in outputs:
                    for sample_id in range(len(output.outputs)):
                        # HACK: filter > (voc_size+specidal_token_num) token_ids, 151664 for qwen model
                        _token_ids = output.outputs[sample_id].token_ids
                        filtered_token_ids = [token_id for token_id in _token_ids if token_id <= 151664]
                        if 151645 not in filtered_token_ids: 
                            # replace the last token with <|im_end|> if no <|im_end|> in response,
                            # this is to ensure successful execution of get_final_eos_mask in multi-turn scenario
                            filtered_token_ids[-1] = 151645

                        response.append(filtered_token_ids)

                # attach model responses to vllm_inputs
                assert len(to_generate)==len(response)

                idx_to_remove = []
                for i_gen, response_ in zip(to_generate, response): 
                    # update conversation
                    response_ = list(response_)                
                    vllm_inputs[i_gen]['prompt_token_ids'] += response_
                    multi_turn_response_mask[i_gen].append(torch.ones(len(response_), dtype=attention_mask.dtype, device=attention_mask.device)) # ASSISTANT, Mark as 1
                    
                    # TODO: determine whether to call search tool, this should be more robust
                    decoded_resp_ = self.tokenizer.decode(response_,skip_special_tokens=True)
                    # if not "<search>" in decoded_resp_:
                    if not decoded_resp_.endswith("<search><img></search>"): # stricter search tigger
                        idx_to_remove.append(i_gen)
                        # NOTE: to_generate.remove(i_gen) # DO NOT .remove() in for loop

                    # print(f"[Round #{current_iteration}] i_gen: {i_gen} | resp: {self.tokenizer.decode(response_)}")

                # update 'to_generate'
                for x in idx_to_remove:
                    to_generate.remove(x)

                print(f"[Round #{current_iteration} Rollout END] For NEXT round, We hava {len(to_generate)} trajs to complete ...")

                # Call Search Tool for some trajs and Get search results
                search_result = []
                for i_todo in tqdm(to_generate, desc=f"[Round #{current_iteration} Searching Progress]"):
                    print(f"[Round #{current_iteration} Search START] Call search tool with {non_tensor_batch['image_urls'][i_todo]} ...")
                    
                    ##########################################################
                    ######### [!!!] Implement Your Search Tool Here ##########
                    ######### [!!!] Implement Your Search Tool Here ##########
                    ######### [!!!] Implement Your Search Tool Here ##########
                    ## Here we use image_url to search relevant information ##
                    ##########################################################
                    image_url_to_search = non_tensor_batch["image_urls"][i_todo]
                    tool_returned_str = call_image_search(image_url_to_search, current_iteration)
                    
                    print(f"[Round #{current_iteration} Search END] Search tool with {non_tensor_batch['image_urls'][i_todo]} returned\n {tool_returned_str}")
                    search_result.append(tool_returned_str)
            
                # Process search results
                to_generate_ = to_generate.copy() # make a copy since we will be modifying to_generate
                assert len(to_generate_) == len(search_result)
                for i_gen_, search_result_ in zip(to_generate_, search_result):
                    
                    # construct next-round prompt
                    search_result_message = 'Searched results: <information>' + search_result_ + '</information>' + " Based on the searched results, please analyze the searched results and continue reasoning inside <reason> and </reason> and give your answer. You MUST keep your answer definitive, short, and clear inside <answer> and </answer> by using words instead of sentences."
                    search_result_message = "<|im_start|>user\n" + search_result_message + "<|im_end|>\n<|im_start|>assistant\n"
                    next_turn_prompt_ids = self.tokenizer.encode(search_result_message)

                    # update conversation
                    vllm_inputs[i_gen_]['prompt_token_ids'] += next_turn_prompt_ids # this might go over response length, but we will cut it later by 'max_response_length_total'
                    multi_turn_response_mask[i_gen_].append(torch.zeros(len(next_turn_prompt_ids),dtype=attention_mask.dtype,device=attention_mask.device)) # USER, Mark as 0
                
                # update pbar
                pbar.update(worker_trajs_count-len(to_generate))

                # update iteration count
                current_iteration += 1

        # re-build response
        response = [] # B'*R, torch.Tensors with unequal lengths
        response_generation_mask = [] # B'*R, torch.Tensors with unequal lengths but align with 'response'
        for i_ in range(batch_size*n):
            # for each traj, we skip first-round prompt_ids/attention_mask
            first_round_prompt_length = prefix_prompt_lengths[i_]
            all_response = torch.tensor(vllm_inputs[i_]['prompt_token_ids'][first_round_prompt_length:], device=idx.device,dtype=idx.dtype)
            response.append(all_response)
            all_response_masks = torch.cat(multi_turn_response_mask[i_][1:], dim=0)
            response_generation_mask.append(all_response_masks) # at least we have single-turn conversation
            assert response[i_].shape[0] == response_generation_mask[i_].shape[0], "shape mismatched between resp_id and resp_mask!"
        assert len(response)==len(response_generation_mask), "length mismatched between response and response_generation_mask!"

        # attention_mask:       prompt           response
        #                 [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        response = pad_to_max_stack(response, self.pad_token_id, dim=0) # Tensor, (B'*R, padded_length), padded_length is the max length of samples in list
        response_generation_mask = pad_to_max_stack(response_generation_mask, 0, dim=0) # Tensor, (B'*R, padded_length)
        assert all([response.size(dim)==response_generation_mask.size(dim) for dim in range(response.ndim)])

        # cut or pad to max length
        # all should be (B*R, self.config.response_length)
        if response.shape[1] > self.config.response_length_total:
            response = response[:,:self.config.response_length_total]
            response_generation_mask = response_generation_mask[:,:self.config.response_length_total]
        elif response.shape[1] < self.config.response_length_total:
            response = pad_sequence_to_length(response, self.config.response_length_total, self.pad_token_id)
            response_generation_mask = pad_sequence_to_length(response_generation_mask, self.config.response_length_total, 0)

        # All for 1st USER prompt
        if self.config.n > 1 and do_sample:
            idx = _repeat_interleave(idx, self.config.n) # (B, max_prompt_length) -> (B*R, max_prompt_length)
            attention_mask = _repeat_interleave(attention_mask, self.config.n)
            position_ids = _repeat_interleave(position_ids, self.config.n)
            batch_size = batch_size * self.config.n
            if 'multi_modal_inputs' in non_tensor_batch.keys():
                non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'], self.config.n)
            # we also need to repeat 'input_prompt_generation_mask'
            input_prompt_generation_mask = _repeat_interleave(input_prompt_generation_mask, self.config.n) # (B, max_prompt_length) -> (B*R, max_prompt_length), all 0

        seq = torch.cat([idx, response], dim=-1) # (B*R, max_prompt_length+max_response_length_total)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        response_attention_mask = get_final_eos_mask(response_id=response, eos_token=[151645], dtype=attention_mask.dtype) # HACK: for qwen, |im_end| is 151645
        # attention_mask: (...,0,0,0,1,1,1), response_attention_mask: (1,1,1,0,0,0,...)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        multi_turn_response_mask = torch.cat([input_prompt_generation_mask, response_generation_mask], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx.contiguous(),
                'responses': response.contiguous(),
                'input_ids': seq.contiguous(),  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask.contiguous(),
                'position_ids': position_ids.contiguous(),
                'multi_turn_response_mask': multi_turn_response_mask.contiguous()
            },
            batch_size=batch_size
        )

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
