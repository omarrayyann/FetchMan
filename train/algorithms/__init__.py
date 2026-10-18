from algorithms.flow_policy import FlowPolicy
from algorithms.ema_callback import EMAWeightAveraging

agents = dict(flow_policy=FlowPolicy)

__all__ = ["FlowPolicy", "EMAWeightAveraging", "agents"]
