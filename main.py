'''
Projeto 2 - Sistemas Inteligentes
MLP - Predict credit card frauds
Sarah R. L. Carneiro
'''

import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from confusion_matrix_pretty_print import plot_confusion_matrix_from_data


class MLP:

    def __init__(self, learningRate, numberOfTimes, hiddenLayersNeurons, valRatio, n_iter_no_change):

        self.learningRate = learningRate  # taxa de apredisagem
        self.numberOfTimes = numberOfTimes  # numero de epocas
        self.hiddenLayersNeurons = hiddenLayersNeurons # número de neurônios por camada escondida
        self.valRatio = valRatio  # percentual dos vetores de treinamento que serão utilizados para validação
        self.n_iter_no_change = n_iter_no_change # critério para early stopping

    def trainMLP(self, X, T):

        return MLPClassifier(
            activation="logistic", # função de ativação = sigmoide
            solver="adam", # gradiente decendente
            hidden_layer_sizes=self.hiddenLayersNeurons,
            max_iter=self.numberOfTimes,
            early_stopping=True,
            learning_rate_init=self.learningRate,
            validation_fraction=self.valRatio,
            n_iter_no_change=self.n_iter_no_change, # o número máximo de épocas sem melhora
            shuffle=True,
            random_state=999,
            verbose=True
        ).fit(X, T)

    # Dataset disponível em https://www.neuraldesigner.com/learning/examples/credit-card-fraud#DataSet
    @staticmethod
    def loadData():

        dataFrame = pd.read_csv('creditcard-fraud.csv', sep=';')

        # retorna o dataset embaralhado
        return dataFrame.sample(frac=1)

    # Adequa os padrões de entrada
    @staticmethod
    def dataTreatment(inputs):

        inputs['is_declined'] = inputs['is_declined'].replace(['yes', 'no'], [1, 0])
        inputs['foreign_transaction'] = inputs['foreign_transaction'].replace(['yes', 'no'], [1, 0])
        inputs['high_risk_countries'] = inputs['high_risk_countries'].replace(['yes', 'no'], [1, 0])

        # normaliza os valores de entrada para valores reais no intervalo [0,1]
        min_max_scaler = preprocessing.MinMaxScaler()
        inputs = min_max_scaler.fit_transform(inputs)

        return inputs

    # Adequa os alvos
    @staticmethod
    def labelTreatment(T):

        return T.replace(['fraudulent', 'non-fraudulent'], [1, 0]).values

def main():

    print('\n\n===== MLP - Predict credit card frauds =====\n\n')
    learningRate = float(input('Insira a taxa de aprendizagem:'))
    percentTrain = float(input('Insira a porcentagem de dados que serão utililizados no treinamento:'))
    percentVal = float(input('Insira a porcentagem de dados que serão utililizados para validação:'))
    numberOfTimes = int(input('Insira o número máximo de épocas:'))
    n_iter_no_change = int(input('Insira o número máximo de épocas sem melhora:'))

    # Carrega e trata os dados
    data = MLP.loadData()
    inputs = MLP.dataTreatment(data.iloc[:, 1:10])
    labels = MLP.labelTreatment(data.iloc[:, 10])
    #print(inputs, inputs.shape)
    #print(labels, labels.shape)

    # cria os conjuntos de treinamento e de teste
    idx = int(percentTrain * len(inputs))
    trainSet = inputs[0:idx, :]
    trainLabels = labels[0:idx]
    testSet = inputs[idx+1:len(inputs), :]
    testLabels = labels[idx+1:len(labels)]

    #print('Tr', trainSet, trainSet.shape)
    #print('Tr L', trainLabels, trainLabels.shape)
    #print('Ts', testSet, testSet.shape)
    #print('Ts L', testLabels, testLabels.shape)

    model = MLP(learningRate, numberOfTimes, [3], percentVal, n_iter_no_change)

    # train
    mlp = model.trainMLP(trainSet, trainLabels)

    # test
    output = mlp.predict(testSet)

    conf = confusion_matrix(testLabels, output)
    print(conf)
    metrics = classification_report(testLabels, output)
    print(metrics)

    plot_confusion_matrix_from_data(testLabels, output)

if __name__ == "__main__":
    main()