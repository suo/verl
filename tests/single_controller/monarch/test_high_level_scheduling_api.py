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

import time

from monarch.actor import endpoint, proc_mesh
from verl.single_controller.monarch.base import (
    MonarchClassWithInitArgs,
    MonarchResourcePool,
    MonarchWorker,
    MonarchWorkerGroup,
)


class TestActor(MonarchWorker):
    # TODO: pass *args and **kwargs is bug prone and not very convincing
    def __init__(self, cuda_visible_devices=None) -> None:
        super().__init__(cuda_visible_devices)

    @endpoint
    def get_node_id(self):
        # Note: Monarch equivalent for node identification
        # This may need to be adapted based on Monarch's actual API
        return "monarch_node_id"


def test():
    # test single-node-no-partition
    print("test single-node-no-partition")
    pm = proc_mesh(gpus=8).get()
    resource_pool = MonarchResourcePool(pm)

    class_with_args = MonarchClassWithInitArgs(cls=TestActor)

    print("create actor worker group")
    actor_wg = MonarchWorkerGroup(
        resource_pool, class_with_args, name_prefix="high_level_api_actor"
    )
    print("create critic worker group")
    critic_wg = MonarchWorkerGroup(
        resource_pool, class_with_args, name_prefix="hight_level_api_critic"
    )
    print("create rm worker group")
    rm_wg = MonarchWorkerGroup(
        resource_pool, class_with_args, name_prefix="high_level_api_rm"
    )
    print("create ref worker group")
    ref_wg = MonarchWorkerGroup(
        resource_pool, class_with_args, name_prefix="high_level_api_ref"
    )

    assert actor_wg.execute_all_sync("get_cuda_visible_devices") == [
        str(i) for i in range(8)
    ]
    assert critic_wg.execute_all_sync("get_cuda_visible_devices") == [
        str(i) for i in range(8)
    ]
    assert rm_wg.execute_all_sync("get_cuda_visible_devices") == [
        str(i) for i in range(8)
    ]
    assert ref_wg.execute_all_sync("get_cuda_visible_devices") == [
        str(i) for i in range(8)
    ]

    del actor_wg
    del critic_wg
    del rm_wg
    del ref_wg

    print("wait 5s for cleanup")
    time.sleep(5)
    # test single-node-multi-partition

    # print("test single-node-multi-partition")
    # rm_pm = proc_mesh(gpus=4).get()
    # ref_pm = proc_mesh(gpus=4).get()
    # rm_resource_pool = MonarchResourcePool(rm_pm)
    # ref_resource_pool = MonarchResourcePool(ref_pm)
    # total_resource_pool = merge_resource_pool(rm_resource_pool, ref_resource_pool)

    # assert rm_resource_pool.world_size == 4
    # assert ref_resource_pool.world_size == 4
    # assert total_resource_pool.world_size == 8

    # actor_wg = MonarchWorkerGroup(total_resource_pool, class_with_args, name_prefix="high_level_api_actor")
    # critic_wg = MonarchWorkerGroup(total_resource_pool, class_with_args, name_prefix="high_level_api_critic")
    # rm_wg = MonarchWorkerGroup(rm_resource_pool, class_with_args, name_prefix="high_level_api_rm")
    # ref_wg = MonarchWorkerGroup(ref_resource_pool, class_with_args, name_prefix="high_level_api_ref")

    # assert actor_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    # assert critic_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(8)]
    # assert rm_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(4)]
    # assert ref_wg.execute_all_sync("get_cuda_visible_devices") == [str(i) for i in range(4, 8)]
