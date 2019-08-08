from protocols.roundtrip_evaluate import roundtrip_evaluate 

def evaluate(**kwargs):
	agent1=kwargs.pop('agents')[0]
	return roundtrip_evaluate(agent1=agent1, agent2=agent1, **kwargs)