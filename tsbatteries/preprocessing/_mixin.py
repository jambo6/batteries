def apply_transform_to_channels(transform):
    """Decorator for transforms that need to be applied along the channels only.

    Assumes the size is (..., length, input_channels), reshapes to (..., input_channels), performs the method
    operation and then reshapes back.

    For example, suppose we wish to use the StandardScaler on a 3d tensor, we can wrap the transform with this
    decorator and it will first reshape to (-1, input_channels), apply the operation, then reshape back.

    Arguments:
        transform (method): An sklearn-type transform method.

    Returns:
        The transformed data in the original shape.
    """

    def inner(self, data):
        # Reshape to 2d
        data_2d = data.reshape(-1, data.size(-1))

        # Apply the transform
        transformed_data = transform(self, data_2d)

        # Reshape back
        output = transformed_data.reshape(data.size())

        return output

    return inner


def apply_fit_to_channels(fit):
    """ See `apply_transform_to_channels`, this function is for the fit method. """

    def inner(self, data, labels=None):
        # Reshape to 2d and return the fit
        data_2d = data.reshape(-1, data.size(-1))
        return fit(self, data_2d, labels=labels)

    return inner
