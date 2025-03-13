from .BesanconModel import BesanconModel

def getGalaxyModel(galaxyModelName, **kwargs):
    if galaxyModelName == "Besancon":
        return BesanconModel(**kwargs)() # an instance of the specific GalaxyModel 
    else:
        print("Model not configured")
    # TODO: implement other models
