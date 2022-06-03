import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import autokeras as ak
from sklearn.metrics import accuracy_score
from typing import Optional


class ClinicalDataModel:
    def __init__(self, general_train_x: pd.DataFrame, general_train_y: pd.DataFrame,
                 specific_train_x: pd.DataFrame, specific_train_y: pd.DataFrame,
                 test_x: pd.DataFrame, test_y: pd.DataFrame,
                 n_models=5):
        self.general_train_x = general_train_x
        self.general_train_y = general_train_y
        self.specific_train_x = specific_train_x
        self.specific_train_y = specific_train_y
        self.test_x = test_x
        self.test_y = test_y
        self.n_models = n_models
        self.pretrained_optimizer = None
        self.meta_trained_optimizer = None
        self.train_accuracy = None
        self.model = None
        self._build_model()

    @property
    def train_x(self):
        return self.general_train_x

    @property
    def train_y(self):
        return self.general_train_y

    def __str__(self):
        return """
        This class stores the data required for Model Agnostic Meta-Learning
        
        There are 3 types of data sets. 
        
        general_train_x/y: The large pool of non-specific training data, i.e., a whole bunch of clinical trials.
        specific_train_x/y: The one clinical trial dataset that you're interested in.
        test_x/y: Some held out data from the clinical trial you're interested in. 
        
        Autokeras is used to build a generic neural network. Autokeras figures out how many layers to use
        and how to parameterize each layer. It's not the best, but it's fine for now.
        
        """

    def _build_model(self):

        # Use AutoKeras to build and evaluate n models
        clf = ak.StructuredDataClassifier(overwrite=True, max_trials=self.n_models)

        # Fit the "best" model found by autokeras
        tb_callback = keras.callbacks.TensorBoard(log_dir='./pretrain_log')
        clf.fit(x=self.general_train_x, y=self.general_train_y, epochs=10, verbose=0,
                validation_data=(self.test_x, self.test_y),
                callbacks=[tb_callback],
                )

        # Evaluate the training accuracy of the chosen model
        predicted_y = clf.predict(self.test_x, verbose=0)
        self.train_accuracy = accuracy_score(self.test_y, predicted_y)

        # Export the "best" model
        self.model = clf.export_model()


class TabularMAML:
    def __init__(self, clinical_data: ClinicalDataModel, num_epochs:Optional[int]=1000):
        self.clinical_data = clinical_data
        self.alpha = 0.001
        self.beta = 0.001
        self.num_epochs = num_epochs
        self.n_chunks = 5
        self._optimizer = keras.optimizers.SGD
        self._loss_func = keras.losses.BinaryCrossentropy()
        self.train_accuracy = None
        self.test_accuracy = None
        self.intermediate_model = None
        self.loss_per_epoch = []
        self._execute_meta_learning()
        self._execute_transfer_learning()

    def __str__(self):
        return """
        This class implements Model Agnostic Meta-Learning
        https://arxiv.org/pdf/1703.03400.pdf
        
        Input: ClinicalDataModel object instantiated with tabular data
        
        """

    def _execute_meta_learning(self):

        # Initialize meta-update optimizer
        optimizer_meta_update = self._optimizer(learning_rate=self.beta)

        # Break up data into chunks
        index_chunk_list = np.array_split(self.clinical_data.train_x.index, self.n_chunks)
        current_model = keras.models.clone_model(self.clinical_data.model)

        for epoch in range(self.num_epochs):

            # Store weights of current model (theta)
            theta = np.array(current_model.get_weights(), dtype=object)
            gradient_sum = self._meta_learn_step(current_model, theta, index_chunk_list)

            # Update current_model weights with the gradient
            optimizer_meta_update.apply_gradients(zip(gradient_sum,
                                                      current_model.trainable_weights))

            # Forward propagate meta-updated model to compute loss in theta' space
            with tf.GradientTape() as tape:
                # Forward propagate model with theta' weights
                current_loss = current_model(self.clinical_data.train_x)
                # Compute loss w.r.t. theta'
                loss_val = self._loss_func(self.clinical_data.train_y, current_loss)
                self.loss_per_epoch.append(loss_val)

        self.intermediate_model = current_model

    def _execute_transfer_learning(self):
        # Make local copy of model and set all layers to "trainable"
        base_model = keras.models.clone_model(self.intermediate_model)
        base_model.trainable = True

        # Find the second to last dense layer
        last_dense_layer = [it for it, layer in enumerate(base_model.layers) if isinstance(layer, keras.layers.Dense)][
            -2]
        for layer in base_model.layers[:last_dense_layer]:
            layer.trainable = False
        model = keras.models.clone_model(base_model)
        tb_callback = tf.keras.callbacks.TensorBoard('./transfer_learning_logs', update_freq=1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               tf.keras.metrics.FalseNegatives(),
                               tf.keras.metrics.AUC(),
                               tf.keras.metrics.Precision(),
                               tf.keras.metrics.Recall(),
                               tf.keras.metrics.TruePositives(),
                               tf.keras.metrics.TrueNegatives(),
                               tf.keras.metrics.FalsePositives(),
                               tf.keras.metrics.FalseNegatives(),
                               ],
                      )

        model.fit(
            x=self.clinical_data.specific_train_x,
            y=self.clinical_data.specific_train_y,
            batch_size=None,
            epochs=500,
            verbose="auto",
            callbacks=[tb_callback],
            validation_split=0.0,
            # validation_data=(maml.clinical_data.test_x, maml.clinical_data.test_y),
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=10,
            validation_steps=10,
            validation_batch_size=None,
            validation_freq=10,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )

        model.evaluate(x=self.clinical_data.test_x,
                       y=self.clinical_data.test_y)

    def _meta_learn_step(self, current_model, theta, index_chunk_list):
        gradient_sum = None
        for batch_num, index_chunk in enumerate(index_chunk_list):

            # Execute one iteration of meta-learning (with theta' weights)
            meta_updated_model = self._model_update_step(current_model, index_chunk)

            # Forward propagate meta-updated model to compute loss in theta' space
            with tf.GradientTape() as tape:
                # Forward propagate model with theta' weights
                meta_update_prediction = meta_updated_model(self.clinical_data.train_x)
                # Compute loss w.r.t. theta'
                loss_val = self._loss_func(self.clinical_data.train_y, meta_update_prediction)

            # Reset the model weights back in theta to prepare for the gradient calculation
            meta_updated_model.set_weights(theta)

            # Compute gradient w.r.t. theta of loss function w.r.t. theta'. The reason I reset meta_updated_model
            # weights to theta instead of using current_model, which has theta weights, is because tf is
            # very picky about gradients and this is the simplest way to make it work.
            gradient_f_theta_prime = tape.gradient(loss_val, meta_updated_model.trainable_weights)

            if batch_num == 0:
                gradient_sum = gradient_f_theta_prime
            else:
                gradient_sum += gradient_f_theta_prime
        return gradient_sum

    def _model_update_step(self, current_model, index_batch):
        train_x = self.clinical_data.train_x.loc[index_batch]
        train_y = self.clinical_data.train_y.loc[index_batch]
        # test_x = self.clinical_data.test_x
        # test_y = self.clinical_data.test_y

        # Make a deep copy of the model object. We're doing this in a parallel map, so we
        # need to be careful not to have multiple workers operating on the same object.
        model = keras.models.clone_model(current_model)

        # Instantiate the optimizer with the 'alpha' parameter from MAML Algorith 2, Line 6.
        optimizer_pre_update = self._optimizer(learning_rate=self.alpha)

        # Prepare to compute gradient of loss function w.r.t. theta. MAML Algorithm 2, line 5.
        # Move model forward and "tape"/record what happens, so detailed results can be referenced.
        with tf.GradientTape() as tape:
            # Forward propagate the model. This does not update weights.
            forward_prop_pre_update = model(train_x)
            # Compute loss function from one step
            pre_update_loss = self._loss_func(train_y, forward_prop_pre_update)

        # MAML Algorith 2, Line 5.
        # Stop recording / exit context manager. Reference "tape" to compute the gradient.
        gradient_f_theta = tape.gradient(pre_update_loss, model.trainable_weights)

        # This line is tricky. Update 'model' by applying gradient. Now f(theta) --> f(theta').
        # Note. The parameter 'Alpha' is accounted for by the optimizer as the learning_rate.
        optimizer_pre_update.apply_gradients(zip(gradient_f_theta, model.trainable_weights))

        return model

    @property
    def leep(self):
        """
        A Measure to evaluate the transferability of learned representations

        Value typically ranges from (-4, -0.5). Larger values (closer to 0) are better.

        https://arxiv.org/pdf/2002.12462.pdf
        """
        prediction_df = self.clinical_data.test_y
        n = prediction_df.shape[0]
        prediction_df.reset_index(inplace=True, drop=True)
        prediction_df['source_labels'] = y_test['clinical_benefit']
        prediction_df.drop('clinical_benefit', axis=1, inplace=True, errors="ignore")
        prediction_df['probability=1'] = pd.Series(self.intermediate_model.predict(self.clinical_data.test_x).T[0],
                                                   name='probability=1')
        prediction_df['probability=0'] = 1 - prediction_df['probability=1']
        prediction_df['target_labels'] = prediction_df['probability=1'].apply(lambda x: np.round(x, 0))
        prediction_df['pair_str'] = prediction_df.apply(lambda row: ''.join(
            [row['source_labels'].astype(int).astype(str), row['target_labels'].astype(int).astype(str)]), axis=1)

        # Compute the empirical joint distribution
        joint_dist_dict = dict()
        label_set = ['0', '1']
        conditional_pairs = ['00', '01', '10', '11']
        for pair in conditional_pairs:
            pair_df = prediction_df[prediction_df['pair_str'] == pair]
            joint_dist_dict[pair] = pair_df['probability={}'.format(pair[0])].sum() / n

        # Compute the empirical marginal distribution
        marginal_dist_dict = dict()
        for y in label_set:
            marginal_dist_dict[y] = sum([joint_dist_dict[_] for _ in conditional_pairs if _[-1] == y])

        # Compute the empirical conditional distribution
        conditional_dist = {key: val / marginal_dist_dict[key[-1]] for key, val in joint_dist_dict.items()}

        outer_sum = 0
        for i in range(n):
            inner_sum = 0
            for z in label_set:
                source_label = prediction_df.loc[i]['source_labels'].astype(int).astype(str)
                inner_sum += conditional_dist[''.join([source_label, str(z)])] * prediction_df.loc[i][
                    'probability={}'.format(z)]
            outer_sum += np.log(inner_sum)

        # "From its definition, the LEEP measure is always negative and larger values
        # (i.e., smaller absolute values) indicate better transferability."
        # LEEP scores tend to range from (-4, -0.5).
        leep_value = outer_sum / n
        return leep_value


def generate_fake_datasets(x_df, y):
    x_new = []
    y_new = []
    mu = 0
    sigma = 0.01
    for i in range(5):
        train_noise = pd.DataFrame(np.random.normal(mu, sigma, x_df.shape), columns=x_df.columns)
        x_new.append(x_df.reset_index(drop=True) + train_noise)
        y_new.append(y.sample(frac=1).reset_index(drop=True))
    x_set = pd.concat(x_new)
    y_set = pd.concat(y_new)
    return x_set, y_set


def oversample(x_df, y):
    from imblearn.over_sampling import SMOTE
    over_sampled = SMOTE(sampling_strategy=lambda y: {0.0: 5000, 1.0: 5000})
    x_synthetic, y_synthetic = over_sampled.fit_resample(x_df, y)
    return x_synthetic, y_synthetic


if __name__ == "__main__":
    # Load prepared tabular datasets
    X_train = pd.read_csv('X_train.csv', index_col=0)
    y_train = pd.read_csv('y_train.csv', index_col=0)

    X_test = pd.read_csv('X_test.csv', index_col=0)
    y_test = pd.read_csv('y_test.csv', index_col=0)

    # Generate synthetic training datasets
    # synthetic_train_x, synthetic_train_y = generate_fake_datasets(X_train, y_train)
    synthetic_train_x, synthetic_train_y = oversample(X_train, y_train)

    # Construct object to store data and auto-build basic keras model
    clinical_data_model = ClinicalDataModel(synthetic_train_x, synthetic_train_y,
                                            X_train, y_train,
                                            X_test, y_test)

    maml = TabularMAML(clinical_data_model)
