import numpy as np
import matplotlib.pyplot as plt
from MomentumMLFFN import MLFFNN
from main import read_data
import multiprocessing

np.random.seed(7246325)
runs = 20
num_tests = 3
seeds = []
epochs = [600, 400, 400]
learning_rate = [0.0008125, 0.0008125, 0.003]
momentum = [0.85, 0.65, 0.85]
def train_network(args):
    input_size, hidden_layers, hidden_sizes, output_size, features, labels, num_epochs, lr, momentum, loss, features_test, labels_test, idx, seed = args
    nn = MLFFNN(input_size, hidden_layers, hidden_sizes, output_size, features, labels, num_epochs, lr, momentum, loss, features_test, labels_test, idx, seed)
    nn.train(idx)
    return idx, nn

def split_into_custom_parts(features, labels):
    # Separate good and bad motor indices
    good_motors = np.where(labels == 1)[0]
    bad_motors = np.where(labels == 0)[0]

    # Calculate the size of each third for good and bad motors
    good_third = len(good_motors) // 3
    bad_third = len(bad_motors) // 3

    # Split good and bad motors into thirds
    good_splits = [
        good_motors[:good_third],
        good_motors[good_third:2 * good_third],
        good_motors[2 * good_third:]
    ]
    bad_splits = [
        bad_motors[:bad_third],
        bad_motors[bad_third:2 * bad_third],
        bad_motors[2 * bad_third:]
    ]

    trainFeaturesPermutations = []
    testFeaturesPermutations = []
    trainLabelsPermutations = []
    testLabelsPermutations = []

    for i in range(3):
        # Combine two thirds for training, one third for testing
        train_indices = np.concatenate([
            good_splits[i], good_splits[(i + 1) % 3],
            bad_splits[i], bad_splits[(i + 1) % 3]
        ])
        test_indices = np.concatenate([good_splits[(i + 2) % 3], bad_splits[(i + 2) % 3]])

        # Shuffle the indices
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        # Create the splits
        trainFeaturesPermutations.append(features[train_indices])
        testFeaturesPermutations.append(features[test_indices])
        trainLabelsPermutations.append(labels[train_indices])
        testLabelsPermutations.append(labels[test_indices])

    return trainFeaturesPermutations, testFeaturesPermutations, trainLabelsPermutations, testLabelsPermutations

def main():
    counter = 0
    for i in range(runs):
        seeds.append(np.random.randint(0, 100000))
    train_args = []
    for i in range(runs):
        labels, features = read_data("L30fft1000.out")
        big_labels, big_features = read_data("L30fft1000.out")
        labels2, features2 = read_data("L30fft150.out")
        for j in range(len(labels2)):
            features2[j] = np.array(features2[j], dtype=float)  # Convert to NumPy array
            features2[j] = features2[j] / np.linalg.norm(features2[j])  # Normalize the vector
            big_features[j].extend(features2[j].tolist())
        np.random.seed(seeds[i])
        shuffled_indices = np.random.permutation(len(labels))
        labels = np.array(labels, dtype=int)[shuffled_indices]
        features = np.array(features, dtype=float)[shuffled_indices]
        big_labels = np.array(big_labels, dtype=int)[shuffled_indices]
        big_features = np.array(big_features, dtype=float)[shuffled_indices]
        labels2 = np.array(labels2, dtype=int)[shuffled_indices]
        features2 = np.array(features2, dtype=float)[shuffled_indices]
        trainFeaturesPermutations, testFeaturesPermutations, trainLabelsPermutations, testLabelsPermutations = split_into_custom_parts(features, labels)
        bigTrainFeaturesPermutations, bigTestFeaturesPermutations, bigTrainLabelsPermutations, bigTestLabelsPermutations = split_into_custom_parts(big_features, big_labels)
        trainFeatures2Permutations, testFeatures2Permutations, trainLabels2Permutations, testLabels2Permutations = split_into_custom_parts(features2, labels2)
        inputData = [
            [trainFeaturesPermutations, trainLabelsPermutations],
            [bigTrainFeaturesPermutations, bigTrainLabelsPermutations],
            [trainFeatures2Permutations, trainLabels2Permutations]
        ]
        testData = [
            [testFeaturesPermutations, testLabelsPermutations],
            [bigTestFeaturesPermutations, bigTestLabelsPermutations],
            [testFeatures2Permutations, testLabels2Permutations]
        ]
        for j in range(len(inputData)):
            for k in range(len(inputData[j][0])):
                index = j * 60 + (k * len(seeds)) + counter
                train_args.append((
                    len(inputData[j][0][k][0]), 1, [len(inputData[j][0][k][0])], 2, inputData[j][0][k], inputData[j][1][k],
                    epochs[j], learning_rate[j], momentum[j], "MSE", testData[j][0][k], testData[j][1][k], index, seeds[i]
                ))

        counter = (counter + 1) % 20
    train_args.sort(key=lambda x: x[-2])
    max_processes = min(20, multiprocessing.cpu_count())
    neural_networks = []
    def process_result(result):
        _, nn = result
        neural_networks.append(nn)

    with multiprocessing.Pool(processes=max_processes) as pool:
        for result in pool.imap_unordered(train_network, train_args):
            process_result(result)

    neural_networks.sort(key=lambda x: x.idx)

    ###Left off here. 0-19 perm0 data0, 20-39 perm1 data0, 40-59 perm2 data0, 60-79 perm0 data1, 80-99 perm1 data1, 100-119 perm2 data1, 120-139 perm0 hidden2, 140-159 perm1 hidden2, 160-179 perm2 hidden2
    titles1 = ["Data Size = 1000", "Data Size = 150", "Data Size = 1000+150"]
    titles2 = ["Permutation = 0", "Permutation = 1", "Permutation = 2"]
    for i in range(9):
        avgTrainPerformance = [np.zeros(epochs[i//3]) for _ in range(9)]
        avgTestPerformance = [np.zeros(epochs[i//3]) for _ in range(9)]
        avgCorrectTraining = [0 for _ in range(9)]
        avgIncorrectTraining = [0 for _ in range(9)]
        avgCorrectTest = [0 for _ in range(9)]
        avgIncorrectTest = [0 for _ in range(9)]

        for j in range(len(seeds)):
            index = i * 20 + j
            nn = neural_networks[index]

            avgTrainPerformance[i] += nn.trainingProgress
            avgTestPerformance[i] += nn.testPerformance

            correct = 0
            incorrect = 0

            # Calculate correct and incorrect training predictions
            for k in range(len(nn.input_data)):
                input_data = nn.input_data[k]
                raw_output, _ = nn.forward_pass(input_data)
                if np.argmax(raw_output) == nn.labels[k]:
                    correct += 1
                else:
                    incorrect += 1

            # Update the corresponding test's correct/incorrect count
            avgCorrectTraining[i] += correct
            avgIncorrectTraining[i] += incorrect

            correct = 0
            incorrect = 0

            # Calculate correct and incorrect test predictions
            for k in range(len(nn.features_test)):
                input_data = nn.features_test[k]
                raw_output, _ = nn.forward_pass(input_data)
                if np.argmax(raw_output) == nn.labels_test[k]:
                    correct += 1
                else:
                    incorrect += 1

            # Update the corresponding test's correct/incorrect count
            avgCorrectTest[i] += correct
            avgIncorrectTest[i] += incorrect

        # Compute averages over all seeds
        avgTrainPerformance[i] /= len(seeds)
        avgTestPerformance[i] /= len(seeds)
        avgCorrectTraining = [x / len(seeds) for x in avgCorrectTraining]
        avgIncorrectTraining = [x / len(seeds) for x in avgIncorrectTraining]
        avgCorrectTest = [x / len(seeds) for x in avgCorrectTest]
        avgIncorrectTest = [x / len(seeds) for x in avgIncorrectTest]

        # Plot performance
        plt.clf()
        plt.figure()
        plt.plot(avgTrainPerformance[i], label="Training Loss")
        plt.plot(avgTestPerformance[i], label="Test Loss")
        plt.legend()
        title = f"Mean Loss Over 20, {titles2[i % 3]}, {titles1[i // 3]} (Momentum)"
        plt.title(title)
        plt.savefig(f"Graph{i}.png")
        plt.close()

        print(
            f"{i} Correct Training: {avgCorrectTraining[i]}, Incorrect Training: {avgIncorrectTraining[i]}, Correct Test: {avgCorrectTest[i]}, Incorrect Test: {avgIncorrectTest[i]}")



if __name__ == "__main__":
    main()