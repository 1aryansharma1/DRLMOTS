"""Delivered, sample-and-hold telemetry for enhanced contract 4.

No physical state or outage schedule enters this object. The engine delivers
successful periodic node reports, recovery-ready notices and failure notices.
"""
from dataclasses import replace

from drlmots.enhanced_rl_contract import (HostTelemetry, LinkTelemetry, NodeTelemetry,
    ObservationSnapshot, PlacementRequest, QueuedTelemetry, VMTelemetry)
from drlmots.objective_model import number


TELEMETRY_PROTOCOL = "periodic_delivered_reports_with_local_admission_v1"


class ObservedTelemetry:
    def __init__(self, contract, host_nodes):
        self.contract = contract
        self.c = contract.context
        self.host_nodes=dict(host_nodes)
        if set(self.host_nodes)!=set(self.c["hosts"]) or set(self.host_nodes.values())-set(self.c["failures"]):
            raise ValueError("explicit host/node telemetry mapping required")
        self.nodes = {n:NodeTelemetry(n,0,True,0) for n in self.c["failures"]}
        self.vms = {v:VMTelemetry(v,0,None,(),tuple(sorted(s["capacities"].items()))) for v,s in self.c["vms"].items()}
        self.hosts = {h:HostTelemetry(h,0,0) for h in self.c["hosts"]}
        self.traffic = {n:{l:LinkTelemetry(l,0,0,0,0,0) for l in self.c["links"]} for n in self.nodes}
        self.last_ready = {n:0.0 for n in self.nodes}
        self.messages = []

    def receive(self, node, time_ms, *, vms, hosts, traffic):
        """Called only for an actually delivered sensor response; no down-state flag."""
        number("telemetry delivery",time_ms)
        if node not in self.nodes or time_ms<self.nodes[node].reported_at_ms:
            raise ValueError("unknown node or stale telemetry delivery")
        if any(type(v) is not VMTelemetry for v in vms) or any(type(v) is not HostTelemetry for v in hosts) or any(type(v) is not LinkTelemetry for v in traffic):
            raise ValueError("typed delivered reports required")
        expected_vms={v for v,s in self.c["vms"].items() if s["node_id"]==node}
        expected_hosts={h for h,n in self.host_nodes.items() if n==node}
        if len(vms)!=len(expected_vms) or len(hosts)!=len(expected_hosts) or len(traffic)!=len(self.c["links"]) or {v.vm_id for v in vms}!=expected_vms or {h.host_id for h in hosts}!=expected_hosts or {l.link_id for l in traffic}!=set(self.c["links"]):
            raise ValueError("incomplete delivered node telemetry")
        if any(v.reported_at_ms!=time_ms for v in (*vms,*hosts,*traffic)):
            raise ValueError("delivered report timestamp mismatch")
        for view in vms:
            self.vms[view.vm_id]=view
        for view in hosts:
            self.hosts[view.host_id]=view
        self.traffic[node]={l.link_id:l for l in traffic}
        old=self.nodes[node]
        since=old.up_since_ms if old.available is True else time_ms
        self.nodes[node]=NodeTelemetry(node,time_ms,True,since,old.detected_failures)
        self.messages.append(dict(time_ms=time_ms,node_id=node,event="sensor_report"))

    def ready(self,node,time_ms):
        old=self.nodes[node]
        self.last_ready[node]=time_ms
        self.nodes[node]=NodeTelemetry(node,time_ms,True,old.up_since_ms if old.available is True else time_ms,old.detected_failures)
        self.messages.append(dict(time_ms=time_ms,node_id=node,event="service_ready"))

    def detected(self,node,time_ms,failed_at_ms):
        old=self.nodes[node]
        # A delayed notice about an older failure cannot undo a delivered ready notice.
        stale=self.last_ready[node]>=failed_at_ms and self.last_ready[node]>0
        self.nodes[node]=NodeTelemetry(node,time_ms,old.available if stale else False,
                                      old.up_since_ms if stale else None,old.detected_failures+1)
        if not stale:
            for vm,spec in self.c["vms"].items():
                if spec["node_id"]==node:
                    self.vms[vm]=VMTelemetry(vm,time_ms,None,(),tuple(sorted(spec["capacities"].items())))
            for host in (h for h,n in self.host_nodes.items() if n==node):
                self.hosts[host]=HostTelemetry(host,time_ms,None)
            self.traffic[node]={l:LinkTelemetry(l,time_ms,0,0,0,0) for l in self.c["links"]}
        self.messages.append(dict(time_ms=time_ms,node_id=node,event="failure_notice",stale=stale))

    def admitted(self,vm,work):
        """Local sends are known without a sensor response; reserve a waiting slot."""
        old=self.vms[vm]
        if len(old.waiting)>=self.c["vms"][vm]["queue_capacity"]:
            raise RuntimeError("observed queue admission exceeded its limit")
        entry=QueuedTelemetry("upload",work.input_kb,0,work.compute_mi,tuple(sorted(work.required_resources.items())))
        # The sensor timestamp stays unchanged: this is local knowledge, not a new heartbeat.
        self.vms[vm]=replace(old,waiting=old.waiting+(entry,))

    def snapshot(self,now,work,*,arrived,completed,unsuccessful,attempts_used=0,previous_node=None,wait_remaining=0):
        links=[]
        for link in self.c["links"]:
            participating={v["node_id"] for v in self.c["vms"].values() if v["link_id"]==link}
            parts=[self.traffic[n][link] for n in sorted(participating)]
            links.append(LinkTelemetry(link,min((p.reported_at_ms for p in parts),default=now),
                sum(p.upload_flows for p in parts),sum(p.download_flows for p in parts),
                sum(p.remaining_upload_kb for p in parts),sum(p.remaining_download_kb for p in parts)))
        request=PlacementRequest.from_work(work,attempts_used=attempts_used,previous_node_id=previous_node,
                                          recovery_wait_remaining_ms=wait_remaining)
        return ObservationSnapshot(now,arrived,completed,unsuccessful,request,
            tuple(self.vms[v] for v in self.c["vms"]),tuple(self.hosts[h] for h in self.c["hosts"]),
            tuple(links),tuple(self.nodes[n] for n in self.c["failures"]))
