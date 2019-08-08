from protocols.roundtrip_evaluate import roundtrip_evaluate

def evaluate(**kwargs):
    agent1, agent2=kwargs.pop('agents')
    return roundtrip_evaluate(agent1=agent1, agent2=agent2, **kwargs)