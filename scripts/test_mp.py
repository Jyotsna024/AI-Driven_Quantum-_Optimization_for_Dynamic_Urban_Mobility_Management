import multiprocessing as mp
import os
from pathlib import Path
import traci

def worker(i):
    sumo_bin = Path(os.environ.get('SUMO_HOME')) / 'bin' / 'sumo.exe'
    cmd = [str(sumo_bin), '-n', 'sumo_files/hsr_test.net.xml', '--no-step-log', 'true', '--no-warnings', 'true']
    traci.start(cmd)
    traci.simulationStep()
    t = traci.simulation.getTime()
    traci.close()
    return i, t

if __name__ == '__main__':
    with mp.Pool(2) as p:
        res = p.map(worker, [1, 2])
    print('Parallel result:', res)
