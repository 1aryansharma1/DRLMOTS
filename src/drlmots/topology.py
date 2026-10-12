"""Explicit cloud/fog memberships, without guessed counts or capacities."""

from dataclasses import dataclass, replace

from drlmots.entities import Host, ResourceTier, VM


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _references(value, name: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise TypeError(f"{name} must be a tuple or list of identifiers")
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} cannot be empty")
    for item in result:
        _identifier(item, name)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate identifiers in {name}")
    return result


@dataclass(frozen=True)
class AccessPoint:
    access_point_id: str
    end_node_id: str

    def __post_init__(self):
        _identifier(self.access_point_id, "access_point_id")
        _identifier(self.end_node_id, "end_node_id")


@dataclass(frozen=True)
class EndNode:
    end_node_id: str
    fog_node_ids: tuple[str, ...]

    def __post_init__(self):
        _identifier(self.end_node_id, "end_node_id")
        object.__setattr__(self, "fog_node_ids", _references(self.fog_node_ids, "fog_node_ids"))


@dataclass(frozen=True)
class FogNode:
    fog_node_id: str
    host_ids: tuple[str, ...]
    vm_ids: tuple[str, ...]

    def __post_init__(self):
        _identifier(self.fog_node_id, "fog_node_id")
        object.__setattr__(self, "host_ids", _references(self.host_ids, "host_ids"))
        object.__setattr__(self, "vm_ids", _references(self.vm_ids, "vm_ids"))


@dataclass(frozen=True)
class CloudDatacenter:
    datacenter_id: str
    host_ids: tuple[str, ...]
    vm_ids: tuple[str, ...]

    def __post_init__(self):
        _identifier(self.datacenter_id, "datacenter_id")
        object.__setattr__(self, "host_ids", _references(self.host_ids, "host_ids"))
        object.__setattr__(self, "vm_ids", _references(self.vm_ids, "vm_ids"))


def _index(items, kind, key):
    if any(not isinstance(item, kind) for item in items):
        raise TypeError(f"all {key} entries must be {kind.__name__} objects")
    result = {getattr(item, key): item for item in items}
    if len(result) != len(items):
        raise ValueError(f"duplicate {key}")
    return result


@dataclass(frozen=True)
class Topology:
    """Each VM/host belongs to one tier node; VM-to-host mapping is unresolved."""

    hosts: tuple[Host, ...]
    vms: tuple[VM, ...]
    fog_nodes: tuple[FogNode, ...] = ()
    cloud_datacenters: tuple[CloudDatacenter, ...] = ()
    access_points: tuple[AccessPoint, ...] = ()
    end_nodes: tuple[EndNode, ...] = ()

    def __post_init__(self):
        for name in (
            "hosts", "vms", "fog_nodes", "cloud_datacenters", "access_points", "end_nodes"
        ):
            items = getattr(self, name)
            if not isinstance(items, (list, tuple)):
                raise TypeError(f"{name} must be a tuple or list")
            object.__setattr__(self, name, tuple(items))
        self.validate()

    def validate(self) -> None:
        """Recheck references and mutable resources before a run."""
        hosts = _index(self.hosts, Host, "host_id")
        vms = _index(self.vms, VM, "vm_id")
        fog = _index(self.fog_nodes, FogNode, "fog_node_id")
        _index(self.cloud_datacenters, CloudDatacenter, "datacenter_id")
        _index(self.access_points, AccessPoint, "access_point_id")
        ends = _index(self.end_nodes, EndNode, "end_node_id")
        for host in self.hosts:
            host.__post_init__()
        for vm in self.vms:
            replace(vm)  # Validate without replacing the caller's queue.

        assigned_hosts, assigned_vms = set(), set()
        for nodes, tier in (
            (self.fog_nodes, ResourceTier.FOG),
            (self.cloud_datacenters, ResourceTier.CLOUD),
        ):
            for node in nodes:
                for host_id in node.host_ids:
                    if host_id not in hosts:
                        raise ValueError(f"unknown host {host_id!r}")
                    if host_id in assigned_hosts:
                        raise ValueError(f"host {host_id!r} belongs to multiple nodes")
                    assigned_hosts.add(host_id)
                for vm_id in node.vm_ids:
                    if vm_id not in vms:
                        raise ValueError(f"unknown VM {vm_id!r}")
                    if vm_id in assigned_vms:
                        raise ValueError(f"VM {vm_id!r} belongs to multiple nodes")
                    if vms[vm_id].tier is not tier:
                        raise ValueError(f"VM {vm_id!r} has the wrong tier")
                    assigned_vms.add(vm_id)
        if assigned_hosts != set(hosts) or assigned_vms != set(vms):
            raise ValueError("every host and VM must belong to exactly one node")
        for access_point in self.access_points:
            if access_point.end_node_id not in ends:
                raise ValueError(f"unknown end node {access_point.end_node_id!r}")
        for end in self.end_nodes:
            for fog_id in end.fog_node_ids:
                if fog_id not in fog:
                    raise ValueError(f"unknown fog node {fog_id!r}")
