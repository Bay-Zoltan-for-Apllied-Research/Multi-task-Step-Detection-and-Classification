import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


dropout = 0.6


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


def relu(x):
    out = np.maximum(x, 0)
    return out


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def load_dataset():
    train_set_x = np.load(train_data_file)
    train_set_x = np.swapaxes(train_set_x, 0, 1)
    train_set_x = np.swapaxes(train_set_x, 0, 2)
    train_set_x = np.expand_dims(train_set_x, axis=3)
    # train_set_x, min, max = normalize_input(train_set_x, None, None)
    print("Min:", min, "Max:", max)
    train_set_y = np.load(train_y_file)
    train_step = np.load(train_step_file)

    return train_set_x, train_set_y, train_step, min, max


class Model(tf.keras.Model):
    """
    multiclass step detector model
    """
    def __init__(self):
        super(Model, self).__init__()

        # Convolutional Layer #1
        self.conv1 = tf.keras.layers.Conv2D(128, (16, 3), padding='valid', strides=(1, 1),
                                            kernel_initializer='glorot_normal')

        # Flatten tensor into a batch of vectors
        self.flat2 = tf.keras.layers.Flatten()

        # Dense Layer1
        self.dense3 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.drop1 = tf.keras.layers.Dropout(dropout)

        # Dense Layer2
        self.dense4 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.drop2 = tf.keras.layers.Dropout(dropout)

        self.dense5 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.drop3 = tf.keras.layers.Dropout(dropout)

        self.dense6 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.drop4 = tf.keras.layers.Dropout(dropout)

        # Dense INT1
        self.dense7 = tf.keras.layers.Dense(units=1)

        # Dense PROB2
        self.dense8 = tf.keras.layers.Dense(units=3)

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        x = self.conv1(inputs)
        x = self.flat2(x)
        x = tf.nn.relu(x)
        x = self.dense3(x)
        x = self.drop1(x)
        x = self.dense4(x)
        x = self.drop2(x)
        INT = self.dense5(x)
        INT = self.drop3(INT)
        INT = self.dense7(INT)
        PROB = self.dense6(x)
        PROB = self.drop4(PROB)
        PROB = self.dense8(PROB)

        return INT, PROB


def loss_fn(INT, PROB, Y, S):

    """
    Multiclass version:
    INT is the same
    PROB is a 3 class softmax
    S is the step size information, it stays the same
    Y is split into 2 pieces along the 1=axis first 3 are the classes last is the intensity
    """
    Y_PROB, Y_INT = tf.split(Y, [3, 1], axis=1)
    step_size, coordinates = tf.split(S, num_or_size_splits=2, axis=1)

    step_size = tf.reshape(step_size, [-1])
    coordinates = tf.reshape(coordinates, [-1])
    Y_INT = tf.reshape(Y_INT, [-1])
    INT = tf.reshape(INT, [-1])

    INT = tf.nn.relu(INT)
    t1 = tf.constant(0.0)
    d_split_size = tf.constant(1)
    loops, _ = tf.shape(S)
    k, _ = tf.shape(S)
    k = tf.cast(k, tf.float32)

    def cond(t1, Y_INT, INT, k, step_size, coordinates, i, loops):
        return tf.less(i, loops)

    def body(t1, Y_INT, INT, k, step_size, coordinates, i, loops):
        Y_INT_piece = tf.slice(Y_INT, tf.slice(coordinates, [i], [d_split_size]),
                               tf.slice(step_size, [i], [d_split_size]))
        Y_INT_piece = tf.cast(Y_INT_piece, tf.float32)
        INT_piece = tf.slice(INT, tf.slice(coordinates, [i], [d_split_size]), tf.slice(step_size, [i], [d_split_size]))
        INT_piece = tf.cast(INT_piece, tf.float32)

        return [tf.add(t1, tf.square(tf.divide(tf.reduce_sum(tf.subtract((Y_INT_piece), (INT_piece))), k))),
                Y_INT, INT, k, step_size, coordinates, tf.add(i, 1), loops]

    cost_loop_INT, _, _, _, _, _, _, _ = tf.while_loop(cond, body,
                                                       [t1, Y_INT, INT, k, step_size, coordinates, 0, loops],
                                                       parallel_iterations=1, back_prop=True, swap_memory=True)

    cost_INT = tf.divide(cost_loop_INT, tf.cast(loops, dtype=tf.float32))  # divide by number of total steps
    # cost_INT = tf.multiply(cost_INT, INT_weight)

    PROB = tf.cast(PROB, tf.float32)
    Y_PROB = tf.cast(Y_PROB, tf.float32)

    cost_PROB = tf.nn.softmax_cross_entropy_with_logits(labels=Y_PROB, logits=PROB)
    prob_divide = tf.shape(cost_PROB)
    prob_divide = tf.cast(prob_divide, dtype=tf.float32)
    cost_PROB = tf.reduce_sum(cost_PROB)
    cost_PROB = tf.divide(cost_PROB, prob_divide)
    print("PROB cost:", cost_PROB)
    print("INT cost:", cost_INT)
    loss = tf.add(cost_INT, cost_PROB)

    return loss


def train():

    # must enable eager execution first
    tf.enable_eager_execution()

    # hyper parameters
    epochs = 1500
    learning_rate = 0.0005

    is_training = True

    # load data
    train_set_x, train_set_y, train_step, min, max = load_dataset()

    X = tf.constant(train_set_x, dtype=tf.float32)
    Y = tf.constant(train_set_y, dtype=tf.float32)
    S = tf.constant(train_step, dtype=tf.int32)


    # create model
    model = Model()

    # prepare optimizer (switch to ADAM later)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    loss_history = []
    epoch_history = []

    # start training

    for e in range(1, epochs + 1):
        # optimize model
        with tf.GradientTape() as tape:
            INT, PROB = model(X, training=True)
            loss_value = loss_fn(INT, PROB, Y, S)
            print("loss value:", loss_value, "loss values shape", tf.shape(loss_value), "epoch:", e)

        loss_history.append(loss_value.numpy())
        epoch_history.append(e)
        grads = tape.gradient(loss_value, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables), global_step=tf.train.get_or_create_global_step())

        if e % 10 == 0:
            validate(model, plot=False)

    plt.plot(epoch_history, loss_history)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    validate(model, plot=True)

    return


def validate(model, plot):
    def step_calculator(PROBS, INTS, lenght):
        """

        S[) -- Chance a step is detected
        NA -- numpy array of tensor S
        Returns:StepNumber
        """

        def determine_step_type(step_stair):
            step = step_stair[0]
            stair = step_stair[1]
            if stair >= step:
                output = 1
            elif stair < step:
                output = 0

            return output

        maxStep = 400
        step_count = 0
        stair_count = 0
        count = 0
        step_and_stair = [0, 0]  # detection count for one step

        for i in range(lenght):
            prob = PROBS[i, :]
            prob_max = np.argmax(prob)
            int = INTS[i, :]
            int = relu(int)

            if prob_max == 1:  # step
                count = count + int
                step_and_stair[0] += 1
                if count >= maxStep:
                    count = 0
                    output = determine_step_type(step_and_stair)
                    step_and_stair = [0, 0]
                    if output == 0:
                        step_count += 1
                    elif output == 1:
                        stair_count += 1

            elif prob_max == 2:  # stair
                count = count + int
                step_and_stair[1] += 1
                if count >= maxStep:
                    count = 0
                    output = determine_step_type(step_and_stair)
                    step_and_stair = [0, 0]
                    if output == 0:
                        step_count += 1
                    elif output == 1:
                        stair_count += 1

        return stair_count, step_count

    def load_dataset_X():

        train_set_x_step = np.load(test_data_file)
        train_set_x_step = np.swapaxes(train_set_x_step, 0, 1)
        test_set = np.swapaxes(train_set_x_step, 0, 2)

        return test_set

    def process_steps(X, process_type):
        INTS = []
        PROBS = []

        lenght, _, _ = np.shape(X)

        instances = []

        for i in range(lenght):
            instances.append(i)
            instance = X[i, :, :]
            instance = np.expand_dims(instance, axis=2)
            instance = np.expand_dims(instance, axis=0)
            INT, PROB = model(instance, training=False)
            PROB = PROB.numpy()
            INTS.append(INT.numpy())
            PROBS.append(PROB)

        PROBS = np.reshape(PROBS, (lenght, 3))
        INTS = np.reshape(INTS, (lenght, 1))

        print("validation running")

        stair_count, step_count = step_calculator(PROBS, INTS, lenght)

        print("done")
        print("step count: °%s" % process_type, step_count)
        print("stair count: °%s" % process_type, stair_count)
        print("total step count %s" % process_type, (stair_count + step_count))

        if plot:
            plt.plot(instances, PROBS)
            plt.xlabel('Epoch #')
            plt.ylabel('PROB')
            plt.title('%s' % process_type)
            plt.grid(True)
            plt.show()

            plt.plot(instances, INTS)
            plt.xlabel('Epoch #')
            plt.ylabel('INT')
            plt.grid(True)
            plt.show()

    print("starting validation")

    X_step = load_dataset_X()

    process_steps(X_step, process_type="mixed")


def main():

    train()

    return


"""
insert file paths here
"""
test_data_file = r""
train_data_file = r""
train_y_file = r""
train_step_file = r""


if __name__ == '__main__':
    main()
