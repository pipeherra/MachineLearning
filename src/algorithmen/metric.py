

def get_distance(args1, args2, metric_name):
    if len(args1) != len(args2):
        raise AttributeError("Len of args1 and args2 is unequal! args1={}, args2={}".format(len(args1), len(args2)))
    distance = 0
    if metric_name == 'chessboard':
        for i in range(0, len(args1)):
            distance = max(distance, abs(args1[i] - args2[i]))

    if metric_name == 'manhattan':
        summed = 0
        for i in range(0, len(args1)):
            summed += abs(args1[i] - args2[i])
        distance = summed

    if metric_name == 'euclidean':
        summed = 0
        for i in range(0, len(args1)):
            summed += pow((abs(args1[i]-args2[i])), 2)
        distance = pow(summed, 1/2)
    return distance
