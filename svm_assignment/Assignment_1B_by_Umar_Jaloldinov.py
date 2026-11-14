import math


data = [
    ["Low", "Clear", "Light", "Short", "ON_TIME"],
    ["Medium", "Clear", "Light", "Short", "ON_TIME"],
    ["Low", "Clear", "Light", "Long", "ON_TIME"],
    ["Low", "Rain", "Light", "Short", "ON_TIME"],
    ["Medium", "Clear", "Light", "Long", "ON_TIME"],
    ["Medium", "Rain", "Light", "Short", "DELAYED"],
    ["Medium", "Clear", "Heavy", "Long", "DELAYED"],
    ["High", "Clear", "Light", "Long", "DELAYED"],
    ["Low", "Rain", "Heavy", "Long", "DELAYED"],
    ["High", "Clear", "Heavy", "Short", "DELAYED"],
    ["High", "Rain", "Heavy", "Long", "SEVERE_DELAY"],
    ["High", "Rain", "Heavy", "Short", "SEVERE_DELAY"],
    ["Medium", "Rain", "Heavy", "Long", "SEVERE_DELAY"],
    ["High", "Rain", "Light", "Long", "SEVERE_DELAY"],
    ["High", "Clear", "Heavy", "Long", "SEVERE_DELAY"]
]

features = [
    "Traffic",
    "Weather",
    "Courier Load",
    "Distance"
]


# =========================
# TASK 1
# Entropy and Information Gain
# =========================

def entropy(rows):
    class_counts = {}

    for row in rows:
        target = row[-1]

        if target not in class_counts:
            class_counts[target] = 0

        class_counts[target] += 1

    total = len(rows)
    result = 0.0

    for count in class_counts.values():
        probability = count / total
        result -= probability * math.log2(probability)

    return result


def split_by_feature(rows, feature_index):
    subsets = {}

    for row in rows:
        value = row[feature_index]

        if value not in subsets:
            subsets[value] = []

        subsets[value].append(row)

    return subsets


def information_gain(rows, feature_index):
    total_entropy = entropy(rows)
    subsets = split_by_feature(rows, feature_index)

    weighted_entropy = 0.0

    for subset in subsets.values():
        weight = len(subset) / len(rows)
        weighted_entropy += weight * entropy(subset)

    return total_entropy - weighted_entropy


print("=" * 50)
print("TASK 1")
print("=" * 50)

print("Root Entropy:", entropy(data))

for i in range(len(features)):
    print(features[i], ":", information_gain(data, i))


# =========================
# TASK 2
# Select Root Feature
# =========================

print("\n" + "=" * 50)
print("TASK 2")
print("=" * 50)

for i in range(len(features)):
    gain = information_gain(data, i)
    print(f"{features[i]:15} {gain:.6f}")


def best_feature(rows, available_features):
    best_index = available_features[0]
    best_gain = information_gain(rows, best_index)

    for index in available_features[1:]:
        gain = information_gain(rows, index)

        if gain > best_gain:
            best_gain = gain
            best_index = index

    return best_index


root_feature = best_feature(data, [0, 1, 2, 3])

print("\nRoot Feature:", features[root_feature])


# =========================
# TASK 3
# Build Complete ID3 Tree
# =========================

print("\n" + "=" * 50)
print("TASK 3")
print("=" * 50)


def majority_class(rows):
    counts = {}

    for row in rows:
        target = row[-1]

        if target not in counts:
            counts[target] = 0

        counts[target] += 1

    best_class = None
    best_count = -1

    for target in counts:
        if counts[target] > best_count:
            best_class = target
            best_count = counts[target]

    return best_class


def build_tree(rows, available_features):
    targets = []

    for row in rows:
        if row[-1] not in targets:
            targets.append(row[-1])

    if len(targets) == 1:
        return targets[0]

    if len(available_features) == 0:
        return majority_class(rows)

    selected_feature = best_feature(rows, available_features)

    tree = {
        features[selected_feature]: {}
    }

    subsets = split_by_feature(rows, selected_feature)

    remaining_features = []

    for feature in available_features:
        if feature != selected_feature:
            remaining_features.append(feature)

    for value in subsets:
        subset = subsets[value]

        tree[features[selected_feature]][value] = build_tree(
            subset,
            remaining_features
        )

    return tree


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "-> " + tree)
        return

    feature = next(iter(tree))

    print(indent + feature + ":")

    branches = tree[feature]

    for value in branches:
        print(indent + "  " + str(value))
        print_tree(branches[value], indent + "    ")


all_features = [0, 1, 2, 3]

tree = build_tree(data, all_features)

print_tree(tree)


# =========================
# TASK 4
# Prediction
# =========================

print("\n" + "=" * 50)
print("TASK 4")
print("=" * 50)


def predict(tree, sample):
    current = tree
    path = []

    while isinstance(current, dict):
        feature = next(iter(current))

        feature_index = features.index(feature)
        value = sample[feature_index]

        path.append(feature + "=" + value)

        current = current[feature][value]

    return current, path


test_data = [
    ["Low", "Clear", "Heavy", "Short", "ON_TIME"],
    ["High", "Rain", "Light", "Short", "DELAYED"],
    ["Medium", "Rain", "Heavy", "Short", "DELAYED"],
    ["High", "Clear", "Light", "Short", "DELAYED"],
    ["Medium", "Clear", "Heavy", "Short", "DELAYED"],
    ["High", "Rain", "Heavy", "Long", "SEVERE_DELAY"]
]


predictions = []

for i, sample in enumerate(test_data):
    predicted, path = predict(tree, sample)

    predictions.append(predicted)

    print("\nT" + str(i + 1))
    print("Decision Path:", " -> ".join(path))
    print("Predicted:", predicted)
    print("Actual:", sample[-1])


# =========================
# TASK 5
# Confusion Matrix and Model Check
# =========================

print("\n" + "=" * 50)
print("TASK 5")
print("=" * 50)


classes = [
    "ON_TIME",
    "DELAYED",
    "SEVERE_DELAY"
]


confusion_matrix = []

for i in range(3):
    confusion_matrix.append([0, 0, 0])


for i in range(len(test_data)):
    actual = test_data[i][-1]
    predicted = predictions[i]

    actual_index = classes.index(actual)
    predicted_index = classes.index(predicted)

    confusion_matrix[actual_index][predicted_index] += 1


print("\nConfusion Matrix:")

print(
    f"{'Actual / Predicted':20}"
    f"{'ON_TIME':12}"
    f"{'DELAYED':12}"
    f"{'SEVERE_DELAY':15}"
)

for i in range(3):
    print(
        f"{classes[i]:20}"
        f"{confusion_matrix[i][0]:12}"
        f"{confusion_matrix[i][1]:12}"
        f"{confusion_matrix[i][2]:15}"
    )


correct = 0

for i in range(3):
    correct += confusion_matrix[i][i]

accuracy = correct / len(test_data)

print("\nAccuracy:", accuracy * 100, "%")


most_confused = None
highest_confusion = 0

for i in range(3):
    for j in range(3):
        if i != j and confusion_matrix[i][j] > highest_confusion:
            highest_confusion = confusion_matrix[i][j]
            most_confused = (classes[i], classes[j])


print(
    "Most often confused:",
    most_confused[0],
    "->",
    most_confused[1]
)
