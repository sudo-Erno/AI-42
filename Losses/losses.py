import numpy as np

def MSE(labels, predictions):
    
    if not isinstance(labels, np.ndarray) or not isinstance(predictions, np.ndarray):
        print("labels and predictions must be numpy array.")
        quit()
    
    samples_length = labels.shape[0]

    return np.sum((labels - predictions)**2) / samples_length

def d_MSE(labels, predictions):
    if not isinstance(labels, np.ndarray) or not isinstance(predictions, np.ndarray):
        print("labels and predictions must be numpy array.")
        quit()

    samples_length = labels.shape[0]

    return np.sum((labels - predictions)*2) / samples_length
    

def BinaryCrossEntropy(labels, predictions):
    
    if not isinstance(labels, np.ndarray) or not isinstance(predictions, np.ndarray):
        print("labels and predictions must be numpy array.")
        quit()

    samples_length = labels.shape[0]
    
    loss = -(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

    loss = np.sum(loss) / samples_length

    return loss

def CrossEntropyLoss(labels, predictions):
    # TODO: Understand it better
    
    if not isinstance(labels, np.ndarray) or not isinstance(predictions, np.ndarray):
        print("labels and predictions must be numpy array.")
        quit()

    loss = - np.sum(labels * np.log(predictions))

    return loss

if __name__ == "__main__":

    predictions = [0.9, 0.05, 0.05]
    labels = [1, 0, 0]

    predictions = np.array(predictions)
    labels = np.array(labels)

    x = CrossEntropyLoss(labels, predictions)
    print(x)
