import DDPG

ddpg = DDPG.DDPG('CartpoleStabShort-v0', 60, 15000, 0.99, 1e-4, 500, 0.002, 20, 0.7, 0.25)
ddpg.train()

def load_stabilizing_model_simulation(path_to_actor, path_to_critic):
    ddpg = DDPG.DDPG('CartpoleStabShort-v0', 60, 15000, 0.99, 1e-4, 500, 0.002, 20, 0.7, 0.2)
    ddpg.load_model(path_to_actor, path_to_critic)
    ddpg.simulate(True)

def load_stabilizing_model_RR(path_to_actor, path_to_critic):
    ddpg = DDPG.DDPG('CartpoleStabRR-v0', 60, 15000, 0.99, 1e-4, 500, 0.002, 20, 0.7, 0.2)
    ddpg.load_model(path_to_actor, path_to_critic)
    ddpg.simulate(False)


def load_swingup_model_simulation(path_to_actor, path_to_critic):
    ddpg = DDPG.DDPG('CartpoleSwingShort-v0', 60, 15000, 0.99, 1e-4, 500, 0.002, 20, 0.7, 0.2)
    ddpg.load_model(path_to_actor, path_to_critic)
    ddpg.simulate(True)

