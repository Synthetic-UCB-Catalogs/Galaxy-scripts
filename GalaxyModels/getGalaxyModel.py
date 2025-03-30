from .BesanconModel import BesanconModel


def getGalaxyModel(galaxyModelName, **kwargs):
    if galaxyModelName == "Besancon":
        # an instance of the specific GalaxyModel
        return BesanconModel("Besancon", **kwargs)
    else:
        print("Model not configured")
    # TODO: implement other models
