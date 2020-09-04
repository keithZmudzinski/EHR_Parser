''' Used for the determination of homonym meanings, given context '''

class WSD:
    def __init__(self):
        self.model = self.load_model()

    def get_meaning(self, instances, term, context):
        # placeholder
        return instances[0]

    def load_model(self):
        pass

    def save_model(self):
        pass
