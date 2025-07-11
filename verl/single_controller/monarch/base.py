import os
import socket
import inspect
from typing import Dict, Optional

from monarch._src.actor.actor_mesh import EndpointProperty
from monarch.actor import Actor, endpoint, Future, MonarchContext, port, ProcMesh, send
from verl.protocol import _padding_size_key, DataProto
from verl.utils.device import get_torch_device, get_visible_devices_keyword

from verl.single_controller.base import Worker

from verl.single_controller.base.decorator import (
    Dispatch,
    Execute,
    get_predefined_dispatch_fn,
    get_predefined_execute_fn,
    MAGIC_ATTR,
    register,
)
from verl.single_controller.base.worker_group import (
    ClassWithInitArgs,
    ResourcePool,
    WorkerGroup,
)


# Define a get() with similar semantics as ray.get()
def get(future):
    if isinstance(future, Future):
        return future.get()
    elif isinstance(future, list):
        return [f.get() for f in future]
    else:
        raise ValueError(f"Unknown type {type(future)}")


# ResourcePool mostly maps onto Monarch's ProcMesh concept, providing an
# abstract set of resources to spawn work into.
#
# Key differences that would take some Monarch work to bridge:
# - Ragged process assignment: ResourcePool accepts a List[int] which is number
#   of processes per node, whereas Monarch today assumes a n * m dense array of
#   processes.
# - ResourcePool is more explicit about GPU/accelerator allocation. The interfaces here are
#   somewhat coupled with the Ray PlacementGroup abstraction. Would probably
#   need to be reworked slightly to upstream.
# - Supports splitting/merging/GPU resource sharing.
class MonarchResourcePool(ResourcePool):
    def __init__(self, pm: ProcMesh) -> None:
        # TODO need to retrieve this from the procmesh
        super().__init__([4], 1)
        self.proc_mesh = pm
        # TODO: this needs to map onto process_on_nodes somehow
        self.setup_actor = self.proc_mesh.spawn("setup", SetupActor).get()
        self._master_addr = (
            self.setup_actor.flatten("anon").slice(anon=0).get_hostname.call_one().get()
        )
        self._master_port = 20122
        self.setup_actor.setup_env.call(self._master_addr, self._master_port).get()


class SetupActor(Actor):
    @endpoint  # type: ignore
    def setup_env(self, master_addr: str, master_port: int):
        ctx = MonarchContext.get()
        rank = ctx.point.rank
        world_size = len(ctx.point)
        local_world_size = 8
        local_rank = rank % local_world_size
        env_vars = {
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "LOCAL_WORLD_SIZE": str(local_world_size),
            "LOCAL_RANK": str(local_rank),
        }
        os.environ.update(env_vars)

    @endpoint  # type: ignore
    def get_hostname(self) -> str:
        return socket.gethostname()


class MonarchWorker(Worker, Actor):
    @endpoint
    def _init(self):
        local_rank = os.environ["LOCAL_RANK"]
        get_torch_device().set_device(int(local_rank))

    @endpoint
    def get_cuda_visible_devices(self):
        """Get the CUDA visible devices configuration."""
        import os

        visible_devices = os.environ.get(
            get_visible_devices_keyword().upper(), "not set"
        )
        return visible_devices

    @endpoint
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO_WITH_FUNC)
    def execute_with_func_generator(self, func, *args, **kwargs):
        """Execute a function with function generator dispatch mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        ret_proto = func(self, *args, **kwargs)
        return ret_proto

    @endpoint
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def execute_func_rank_zero(self, func, *args, **kwargs):
        """Execute a function in rank zero execution mode.

        Args:
            func:
                Function to execute
            *args:
                Positional arguments for the function
            **kwargs:
                Keyword arguments for the function
        """
        result = func(*args, **kwargs)
        return result


class MonarchClassWithInitArgs(ClassWithInitArgs):
    pass


def func_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    class Functor:
        def __call__(this, *args, **kwargs):
            args, kwargs = dispatch_fn(self, *args, **kwargs)
            padding_count = kwargs.pop(_padding_size_key, 0)
            output = execute_fn(method_name, *args, **kwargs)
            if blocking:
                output = get(output)
            output = collect_fn(self, output)
            if padding_count > 0:
                if isinstance(output, DataProto):
                    indices = [i for i in range(len(output))][:-padding_count]
                    output = output.select_idxs(indices)
                elif isinstance(output, list):
                    output = output[:-padding_count]
            return output

    # use class type to pass the method_name to get a better observability
    return type(method_name, (Functor,), {})()


# WorkerGroup mostly maps onto ActorMesh.
#
# Key differences:
# - WorkerGroup initializes some torch.distributed related env vars
#   (WORLD_SIZE, RANK, MASTER_ADDR, MASTER_PORT). This is handled by
#   `SetupActor` in this example.
# - Dispatch styles (e.g. broadcast) are coupled with methods definitions.
#   Monarch is more flexible in this respect (you choose at dispatch time).
#   We can emulate their way of doing it, although user-defined dispatchers
#   would need to be written differently.
class MonarchWorkerGroup(WorkerGroup):
    def __init__(
        self,
        resource_pool: MonarchResourcePool,
        class_with_init_args: MonarchClassWithInitArgs,
        name_prefix: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(resource_pool, **kwargs)
        self.class_with_init_args = class_with_init_args
        actor_name = (
            f"{name_prefix}_{class_with_init_args.cls.__name__}"
            if name_prefix
            else class_with_init_args.cls.__name__
        )
        self._actor_mesh = resource_pool.proc_mesh.spawn(
            actor_name,
            class_with_init_args.cls,
            *class_with_init_args.args,
            **class_with_init_args.kwargs,
        ).get()
        self._actor_mesh._init.call().get()
        self._bind_worker_methods()

    @classmethod
    def from_detached(
        cls,
        name_prefix=None,
        worker_names=None,
        worker_handles=None,
        ray_cls_with_init=None,
        **kwargs,
    ):
        raise NotImplementedError("from_detached is not implemented for Monarch")

    @property
    def world_size(self):
        return self._actor_mesh.size()

    def _bind_worker_methods(self):
        user_defined_cls = self.class_with_init_args.cls

        method_names = []
        for method_name in dir(user_defined_cls):
            try:
                method = getattr(user_defined_cls, method_name)
                # Unwrap the Monarch endpoint to get at the original method.
                if isinstance(method, EndpointProperty):
                    method = method._method
                assert callable(
                    method
                ), f"{method_name} in {user_defined_cls} is not callable"
            except Exception:
                # if it is a property, it will fail because Class doesn't have instance property
                continue

            if hasattr(method, MAGIC_ATTR):
                # this method is decorated by register
                attribute = getattr(method, MAGIC_ATTR)
                assert isinstance(
                    attribute, Dict
                ), f"attribute must be a dictionary. Got {type(attribute)}"
                assert (
                    "dispatch_mode" in attribute
                ), "attribute must contain dispatch_mode in its key"

                dispatch_mode = attribute["dispatch_mode"]
                execute_mode = attribute["execute_mode"]
                blocking = attribute["blocking"]

                # get dispatch fn
                if isinstance(dispatch_mode, Dispatch):
                    # get default dispatch fn
                    fn = get_predefined_dispatch_fn(dispatch_mode=dispatch_mode)
                    dispatch_fn = fn["dispatch_fn"]
                    collect_fn = fn["collect_fn"]
                else:
                    assert isinstance(dispatch_mode, dict)
                    assert "dispatch_fn" in dispatch_mode
                    assert "collect_fn" in dispatch_mode
                    dispatch_fn = dispatch_mode["dispatch_fn"]
                    collect_fn = dispatch_mode["collect_fn"]

                # get execute_fn_name
                execute_mode = get_predefined_execute_fn(execute_mode=execute_mode)
                wg_execute_fn_name = execute_mode["execute_fn_name"]

                # get execute_fn from string
                try:
                    execute_fn = getattr(self, wg_execute_fn_name)
                    assert callable(execute_fn), "execute_fn must be callable"
                except Exception:
                    print(f"execute_fn {wg_execute_fn_name} is invalid")
                    raise

                # bind a new method to the RayWorkerGroup
                func = func_generator(
                    self,
                    method_name,
                    dispatch_fn=dispatch_fn,
                    collect_fn=collect_fn,
                    execute_fn=execute_fn,
                    blocking=blocking,
                )

                try:
                    setattr(self, method_name, func)
                    method_names.append(method_name)
                except Exception as e:
                    raise ValueError(f"Fail to set method_name {method_name}") from e
        return method_names

    def _execute_one_rank(self, rank, method_name: str, *args, **kwargs):
        endpoint = getattr(self._actor_mesh, method_name)
        p, r = port(endpoint, once=True)
        send(endpoint, args, kwargs, port=p, selection=rank)
        return r.recv()

    def execute_rank_zero_sync(self, method_name: str, *args, **kwargs):
        """Execute a method on rank zero worker synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method execution
        """
        return get(self.execute_rank_zero_async(method_name, *args, **kwargs))

    def execute_rank_zero_async(self, method_name: str, *args, **kwargs):
        """Execute a method on rank zero worker asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        return self._execute_one_rank(0, method_name, *args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        """Alias for execute_rank_zero_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Remote object reference to the method execution
        """
        return self.execute_rank_zero_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        """Alias for execute_all_async.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        return self.execute_all_async(method_name, *args, **kwargs)

    def execute_all_sync(self, method_name: str, *args, **kwargs):
        """Execute a method on all workers synchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of results from all workers
        """
        return get(self.execute_all_async(method_name, *args, **kwargs))

    def execute_all_async(self, method_name: str, *args, **kwargs):
        """Execute a method on all workers asynchronously.

        Args:
            method_name: Name of the method to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            List of remote object references to the method executions
        """
        # Here, we assume that if all arguments in args and kwargs are lists,
        # and their lengths match len(self._workers), we'll distribute each
        # element in these lists to the corresponding worker
        # print(f"execute_all_async: method {method_name}({args}, {kwargs})")
        length = self._actor_mesh.size()
        if all(isinstance(arg, list) for arg in args) and all(
            isinstance(kwarg, list) for kwarg in kwargs.values()
        ):
            if all(len(arg) == length for arg in args) and all(
                len(kwarg) == length for kwarg in kwargs.values()
            ):
                # print(f"splitting args and kwargs into {length} shards")
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    result.append(
                        self._execute_one_rank(
                            i, method_name, *sliced_args, **sliced_kwargs
                        )
                    )
                return result

        return [
            self._execute_one_rank(i, method_name, *args, **kwargs)
            for i in range(self._actor_mesh.size())
        ]

    def spawn(self, prefix_set):
        """Spawn to a dictionary of worker groups, each with a subset of method with prefix.
        Args:
            prefix_set: Set of prefixes to create worker groups for
        Returns:
            Dictionary of worker groups keyed by prefix
        """

        def _rebind_actor_methods(worker_group, actor_name):
            prefix: str = actor_name + "_"
            for method_name in dir(self):  # Look at original worker group methods
                if method_name.startswith(prefix):
                    # only valid when Python >= 3.9
                    original_method_name = method_name.removeprefix(prefix)
                    method = getattr(self, method_name)
                    setattr(worker_group, original_method_name, method)

        new_worker_group_dict = {}
        for prefix in prefix_set:
            # Create a new worker group that shares the same actor mesh
            new_worker_group = MonarchWorkerGroup.__new__(MonarchWorkerGroup)

            # Copy essential attributes from the original worker group
            # TODO this there a cleaner way to do this?
            new_worker_group.class_with_init_args = self.class_with_init_args
            new_worker_group._actor_mesh = self._actor_mesh  # Share the same actor mesh
            new_worker_group._is_init_with_detached_workers = getattr(
                self, "_is_init_with_detached_workers", False
            )
            new_worker_group.fused_worker_used = getattr(
                self, "fused_worker_used", False
            )
            new_worker_group._procecss_dispatch_config = getattr(
                self, "_procecss_dispatch_config", None
            )

            # Rebind only the methods for this prefix
            _rebind_actor_methods(new_worker_group, prefix)
            new_worker_group_dict[prefix] = new_worker_group
        return new_worker_group_dict


def _bind_workers_method_to_parent(cls, key, user_defined_cls):
    """
    Binds the methods of each worker to the WorkerDict.
    Note that we only bind public methods that are decorated by register
    """
    for method_name in dir(user_defined_cls):
        try:
            method = getattr(user_defined_cls, method_name)
            # Unwrap the Monarch endpoint to get at the original method if needed
            if isinstance(method, EndpointProperty):
                method = method._method
            assert callable(
                method
            ), f"{method_name} in {user_defined_cls} is not callable"
        except Exception:
            # if it is a property, it will fail because Class doesn't have instance property
            continue

        if hasattr(method, MAGIC_ATTR):

            def generate_function(name, key=key):
                def func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    worker = self.worker_dict[key]
                    method_attr = getattr(worker, name)
                    # If it's an EndpointProperty, get the underlying method
                    if isinstance(method_attr, EndpointProperty):
                        actual_method = method_attr._method
                        return actual_method(worker, *args, **kwargs)
                    else:
                        return method_attr(*args, **kwargs)

                async def async_func(self, *args, **kwargs):
                    # dispatch to the actual worker
                    worker = self.worker_dict[key]
                    method_attr = getattr(worker, name)
                    # If it's an EndpointProperty, get the underlying method
                    if isinstance(method_attr, EndpointProperty):
                        actual_method = method_attr._method
                        return await actual_method(worker, *args, **kwargs)
                    else:
                        return await method_attr(*args, **kwargs)

                wrapper = (
                    async_func if inspect.iscoroutinefunction(method) else func
                )  # noqa: B023

                return wrapper

            func = generate_function(method_name)
            # pass MAGIC_ATTR for outer worker group
            attrs = getattr(method, MAGIC_ATTR)
            setattr(func, MAGIC_ATTR, attrs)
            # Add @endpoint decorator to make it callable via Monarch
            func = endpoint(func)
            # prefix method name with key
            try:
                # bind direct rollout method to class without prefix
                if (
                    attrs["dispatch_mode"] == Dispatch.DIRECT_ROLLOUT_METHOD
                    and "rollout" in key
                ):
                    assert not hasattr(
                        cls, method_name
                    ), f"conflict direct rollout method {method_name} with role {key}"
                    setattr(cls, method_name, func)
                    print(f"bind role {key} method {method_name} to class {cls}")
                else:
                    method_name_with_prefix = key + "_" + method_name
                    setattr(cls, method_name_with_prefix, func)
            except Exception as e:
                raise ValueError(f"Fail to set method_name {method_name}") from e


def create_colocated_worker_cls(class_dict: dict[str, MonarchClassWithInitArgs]):
    """
    This function should return a class instance that delegates the calls to every
    cls in cls_dict
    """
    cls_dict = {}
    init_args_dict = {}

    # For Monarch, we'll use MonarchWorker as the base class
    worker_cls = MonarchWorker

    for key, cls in class_dict.items():
        cls_dict[key] = cls.cls
        init_args_dict[key] = {"args": cls.args, "kwargs": cls.kwargs}
    assert cls_dict.keys() == init_args_dict.keys()

    # Create the colocated worker class
    class WorkerDict(worker_cls):
        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                self.worker_dict[key] = user_defined_cls(
                    *init_args_dict[key].get("args", ()),
                    **init_args_dict[key].get("kwargs", {}),
                )

    # now monkey-patch the methods from inner class to WorkerDict
    for key, user_defined_cls in cls_dict.items():
        _bind_workers_method_to_parent(WorkerDict, key, user_defined_cls)

    # Return as MonarchClassWithInitArgs
    return MonarchClassWithInitArgs(cls=WorkerDict)
