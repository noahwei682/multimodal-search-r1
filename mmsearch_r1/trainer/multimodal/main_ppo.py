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
注意：我们没有将main函数与ray_trainer合并，因为ray_trainer被其他main函数使用。
这个文件实现了一个基于PPO(Proximal Policy Optimization)算法的多模态搜索训练器。
"""

import hydra
import ray

from mmsearch_r1.trainer.multimodal.ray_trainer import RayPPOTrainer


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    """
    主入口函数，使用hydra进行配置管理
    Args:
        config: hydra配置对象，包含了训练所需的所有参数
    """
    run_ppo(config)


def run_ppo(config, compute_score=None):
    """
    运行PPO训练的主函数
    Args:
        config: 训练配置
        compute_score: 可选的评分计算函数
    """
    if not ray.is_initialized():
        # 初始化本地ray集群，设置环境变量
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # 确保main_task不会在head节点上调度
def main_task(config, compute_score=None):
    """
    Ray分布式执行的主任务
    Args:
        config: 训练配置
        compute_score: 可选的评分计算函数
    """
    # 打印初始配置
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_to_local

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True 会评估符号值
    OmegaConf.resolve(config)

    # 从HDFS下载checkpoint到本地
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # 实例化tokenizer和processor
    from verl.utils import hf_processor, hf_tokenizer

    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # 用于多模态LLM，可能为None

    # 根据策略类型定义worker类
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        # FSDP (Fully Sharded Data Parallel) 策略
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import CriticWorker
        from mmsearch_r1.workers.multimodal.fsdp_workers import ActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        # Megatron策略
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from mmsearch_r1.trainer.multimodal.ray_trainer import ResourcePoolManager, Role

    # 定义角色到worker的映射关系
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    # 设置资源池配置
    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # 配置奖励模型
    # 我们这里采用多源奖励函数:
    # - 对于基于规则的rm，直接调用奖励分数
    # - 对于基于模型的rm，调用模型
    # - 对于代码相关的prompt，如果有测试用例就发送到沙箱
    # - 最后，将所有奖励组合在一起
    # - 奖励类型取决于数据的标签
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # 选择奖励管理器
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from mmsearch_r1.workers.multimodal.reward import MMSearchR1_RewardManager
        reward_manager_cls = MMSearchR1_RewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError
    
    # 实例化训练和验证用的奖励函数
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    # 注意：验证时总是使用基于函数的RM
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    # 创建资源池管理器
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # 初始化PPO训练器
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )

    trainer.init_workers()

    # 检查多轮对话设置
    if config.actor_rollout_ref.rollout.name == "vllm_multiturn_mmsearch":
        assert (
            config.actor_rollout_ref.actor.use_multi_turn_response_mask == True
        ), "对于多轮对话场景，需要设置`actor_rollout_ref.actor.use_multi_turn_response_mask`为True以在update_policy()中正确精炼`response_mask`"
    else:
        assert (
            config.actor_rollout_ref.actor.use_multi_turn_response_mask == False
        ), "对于非多轮对话场景，需要设置`actor_rollout_ref.actor.use_multi_turn_response_mask`为False"
    
    # 开始训练
    trainer.fit()


if __name__ == '__main__':
    main()
