import gymnasium as gym


def check_space_is_discrete(space: gym.Space) -> bool:
    if isinstance(space, gym.spaces.Discrete):
        return True
    return False


def check_space_is_box(space: gym.Space) -> bool:
    return not check_space_is_discrete(space)
