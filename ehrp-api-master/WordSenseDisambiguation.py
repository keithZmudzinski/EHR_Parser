''' Used for the determination of homonym meanings, given context '''

class WSD:
    def __init__(self, location=None):
        self.location = location
        self.model = self.load_model()

    def get_meaning(self, instances, term, context):
        # placeholder
        return instances[0]

    def load_model(self, path=self.location):
        if self.location:
            return loaded_model
        return None

    def save_model(self, path=self.location):
        pass
