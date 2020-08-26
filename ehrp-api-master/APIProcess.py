import multiprocessing as mp
import ehrp_api

class APIProcess(mp.Process):
    def __init__(self, id):
        mp.Process.__init__(self)
        self.id = id

    def run(self):
        print('Starting process {}'.format(self.id))
        ehrp_api.main()
        print('Finished process {}'.format(self.id))
