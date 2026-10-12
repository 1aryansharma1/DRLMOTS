from dataclasses import replace

import pytest

from drlmots.entities import Host, ResourceTier, VM
from drlmots.topology import AccessPoint, CloudDatacenter, EndNode, FogNode, Topology


def make_topology():
    return Topology(
        hosts=(Host("hf", 3, 32, 1000), Host("hc", 3, 32, 1000)),
        vms=(VM("vf", ResourceTier.FOG, 200), VM("vc", ResourceTier.CLOUD, 200)),
        fog_nodes=(FogNode("fog", ("hf",), ("vf",)),),
        cloud_datacenters=(CloudDatacenter("cloud", ("hc",), ("vc",)),),
        access_points=(AccessPoint("ap", "end"),),
        end_nodes=(EndNode("end", ("fog",)),),
    )


def test_linked_fog_and_cloud_resources():
    topology = make_topology()
    topology.validate()
    assert topology.access_points[0].end_node_id == topology.end_nodes[0].end_node_id
    assert topology.end_nodes[0].fog_node_ids == ("fog",)


@pytest.mark.parametrize("field,value,error", [
    ("hosts", (Host("hf", 3, 32, 1000), Host("hf", 3, 32, 1000)), "duplicate host_id"),
    ("vms", (VM("vf", ResourceTier.FOG, 200), VM("vf", ResourceTier.FOG, 200)), "duplicate vm_id"),
    ("access_points", (AccessPoint("ap", "missing"),), "unknown end node"),
    ("end_nodes", (EndNode("end", ("missing",)),), "unknown fog node"),
    ("fog_nodes", (FogNode("fog", ("hf",), ("missing",)),), "unknown VM"),
    ("fog_nodes", (FogNode("fog", ("missing",), ("vf",)),), "unknown host"),
    ("fog_nodes", (), "exactly one node"),
    ("vms", (VM("vf", ResourceTier.CLOUD, 200), VM("vc", ResourceTier.CLOUD, 200)), "wrong tier"),
])
def test_invalid_relationships(field, value, error):
    with pytest.raises(ValueError, match=error):
        replace(make_topology(), **{field: value})


def test_resource_cannot_belong_to_multiple_nodes():
    with pytest.raises(ValueError, match="multiple nodes"):
        replace(make_topology(), fog_nodes=(
            FogNode("fog", ("hf",), ("vf",)),
            FogNode("fog2", ("hf",), ("vf",)),
        ))


def test_memberships_copy_callers_lists():
    memberships = ["vf"]
    node = FogNode("fog", ["hf"], memberships)
    memberships.append("other")
    assert node.vm_ids == ("vf",)


@pytest.mark.parametrize("memberships", [(), ("vf", "vf"), ("",)])
def test_invalid_membership_list(memberships):
    with pytest.raises(ValueError):
        FogNode("fog", ("hf",), memberships)


def test_mutated_vm_is_revalidated():
    topology = make_topology()
    topology.vms[0].bandwidth_reported = float("nan")
    with pytest.raises(ValueError, match="bandwidth_reported"):
        topology.validate()
