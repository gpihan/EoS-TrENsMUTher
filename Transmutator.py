from src.Parameters import Parameters

if __name__ == "__main__":
    Param = Parameters()
    Param.read_parameters("parameters.py")

    print(Param.dict["edens_min"])
