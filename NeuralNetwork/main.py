from neuralNetwork import NeuralNet

def main():
    print("Criar (0) ou carregar (1) modelo?: ", end='')
    decision = int(input())
    neuralNet = NeuralNet('../tar2_sinais_vitais_treino_com_label.txt')
    isOk = True

    if decision == 0:
        neuralNet.createModel()
        neuralNet.runModel()
    else:
        isOk = neuralNet.loadModel()

    if isOk:
        rowQnt = 1
        while rowQnt != 0:
            print("How many rows do you want to predict?: ", end='')
            rowQnt = int(input())
            if rowQnt != 0:
                neuralNet.predictData(rowQnt)

        print("Do you want to save this model? (1 yes, 0 no): ", end='')
        save = int(input())
        if save == 1:
            neuralNet.saveModel()

    return

main()