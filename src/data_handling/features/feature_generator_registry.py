class FeatureGeneratorRegistry:
    def __init__(self):
        self._generators = {}

    def register(self, generator_class):
        name = generator_class.__name__
        if name in self._generators:
            raise ValueError(f"Feature generator {name} already registered.")
        self._generators[name] = generator_class()

        return generator_class

    def get(self, name):
        return self._generators.get(name)

    def get_all(self):
        return self._generators

    def get_all_names(self):
        return list(self._generators.keys())

feature_generator_registry = FeatureGeneratorRegistry()
