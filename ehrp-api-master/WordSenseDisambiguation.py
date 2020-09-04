''' Used for the determination of homonym meanings, given context '''
import os

class WSD:
    def __init__(self, location=os.path.join('resources', 'WSD_model.txt')):
        self.location = location
        self.model = self.load_model()

    def get_meaning(self, instances, term, context):
        # placeholder
        return instances[0]

    def load_model(self, path=None):
        model_path = path if path else self.location

        if model_path == 'NEW_MODEL':
            return None

        # LOAD MODEL HERE
        return model

    def save_model(self, path=None):
        pass
