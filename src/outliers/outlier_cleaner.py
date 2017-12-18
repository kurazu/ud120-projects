from collections import namedtuple


Result = namedtuple('Result', ('age', 'net_worth', 'error'))


def part(array, factor=0.9):
    return array[:int(0.9 * len(array))]


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    items = (
        Result(age, net_worth, abs(prediction - net_worth))
        for (prediction, age, net_worth) in zip(predictions, ages, net_worths)
    )

    sorted_items = sorted(items, key=lambda r: r.error)
    return part(sorted_items)
