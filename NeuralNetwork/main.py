from neuralNetwork import NeuralNet

def main():
    neuralNet = NeuralNet('../tar2_sinais_vitais_treino_com_label.txt')

    neuralNet.createModel()
    neuralNet.runModel()

    return

main()