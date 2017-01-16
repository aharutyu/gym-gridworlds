from gym.envs.registration import register

register(
    id='FourRooms-v0',
    entry_point='FourRooms.envs:FourRooms',
    timestep_limit=100000,

    #id='absgridworld-v0',
    #entry_point='gridworld.envs:AbstractGridWorld',
    #timestep_limit=1000,
)
