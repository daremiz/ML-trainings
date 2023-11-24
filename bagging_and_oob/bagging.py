#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob

    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            # Your Code Here
            # Randomly sample indices with replacement (bootstrapping)
            indices = np.random.choice(data_length, size=data_length, replace=True)
            # Store the indices for this bag
            self.indices_list.append(indices)

    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        for bag_indices in self.indices_list:
            model = model_constructor()
            data_bag, target_bag = data[bag_indices], target[bag_indices]
            self.models_list.append(model.fit(data_bag, target_bag))  # store fitted models here

        if self.oob:
            self.data = data
            self.target = target

    def predict(self, data):
        '''
        Get average prediction for every object from the passed dataset
        '''
        if not self.models_list:
            raise ValueError("Models have not been fitted. Call 'fit' method first.")

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Make predictions using each model
        predictions = [model.predict(data) for model in self.models_list]

        # Average the predictions
        average_prediction = np.mean(predictions, axis=0)

        return average_prediction

    def _get_oob_predictions_from_every_model(self):
        '''
        Generates a list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during the training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Iterate through each model and its corresponding out-of-bag indices
        for model, bag_indices in zip(self.models_list, self.indices_list):
            oob_indices = np.setdiff1d(np.arange(len(self.data)), bag_indices)
            oob_data = self.data[oob_indices]

            # Make predictions on out-of-bag data and store them in the corresponding lists
            predictions = model.predict(oob_data)
            for i, oob_index in enumerate(oob_indices):
                list_of_predictions_lists[oob_index].append(predictions[i])

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from the training set.
        If the object has been used in all bags during the training phase, return None instead of a prediction.
        '''
        self._get_oob_predictions_from_every_model()
        averaged_oob_predictions = np.empty(len(self.data), dtype=object)
        for i, predictions_list in enumerate(self.list_of_predictions_lists):
            # Check if the object has been used in all bags during the training phase
            if len(predictions_list) == 0:
                averaged_oob_predictions[i] = None
            else:
                # Compute the average prediction for the object
                averaged_oob_predictions[i] = np.mean(predictions_list)
        self.oob_predictions = averaged_oob_predictions

    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        # Filter out objects with None predictions
        valid_indices = [i for i, prediction in enumerate(self.oob_predictions) if prediction is not None]

        if not valid_indices:
            raise ValueError("No objects with valid out-of-bag predictions found.")

        # Extract the valid predictions and corresponding true targets
        valid_predictions = np.array([self.oob_predictions[i] for i in valid_indices])
        valid_targets = self.target[valid_indices]

        # Compute mean square error
        mse = np.mean((valid_predictions - valid_targets) ** 2)

        return mse

