from monarch.actor import endpoint
from verl.single_controller.base.megatron.worker import (
    DistGlobalInfo,
    DistRankInfo,
    MegatronWorker,
)
from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup
from verl.single_controller.monarch import (
    MonarchClassWithInitArgs,
    MonarchResourcePool,
    MonarchWorker,
    MonarchWorkerGroup,
)


class NVMegatronMonarchWorkerGroup(MonarchWorkerGroup, MegatronWorkerGroup):
    """
    MegatronWorkerGroup will query each worker of its megatron rank info and store it inside the WorkerGroup
    so that the dispatcher can use it to dispatch data.
    """

    def __init__(
        self,
        resource_pool: MonarchResourcePool,
        class_with_init_args: MonarchClassWithInitArgs,
        **kwargs,
    ):
        """
        Initialize the NVMegatronRayWorkerGroup.

        Args:
            resource_pool (RayResourcePool): The resource pool containing worker resources
            ray_cls_with_init (RayClassWithInitArgs): The Ray class with initialization arguments
            **kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(
            resource_pool=resource_pool,
            class_with_init_args=class_with_init_args,
            **kwargs,
        )
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(
            method_name="get_megatron_rank_info"
        )
        self._megatron_global_info: DistGlobalInfo = self.execute_rank_zero_sync(
            method_name="get_megatron_global_info"
        )


class MegatronMonarchWorker(MonarchWorker, MegatronWorker):
    @endpoint
    def get_megatron_global_info(self):
        return super().get_megatron_global_info()

    @endpoint
    def get_megatron_rank_info(self):
        return super().get_megatron_rank_info()
