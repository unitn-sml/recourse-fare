from setuptools import setup, find_packages

if __name__ == "__main__":

    setup(
        name='rl_mcts',
        version='0.0.1',
        packages=find_packages(include=["rl_mcts", "rl_mcts.*", "rl_mcts.core.*"]),
        url='',
        license='MIT',
        author='Giovanni De Toni',
        author_email='giovanni.detoni@unitn.it',
        description='Simple package implementing a RL agent trained together with MCTS',
        platforms="any",
    )
