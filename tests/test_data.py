import numpy as np

from q4.data import JOB_FEATURES, generate_training_data, train_and_predict


def test_exact_seeded_data_generation():
    features, target = generate_training_data()
    assert features.shape == (100, 3)
    assert target.shape == (100,)
    assert np.allclose(features[0], [51.081020396059, 4.324176327621, 1.571790130209])
    assert np.isclose(target[0], 3.6425416761996847)
    features_again, target_again = generate_training_data()
    assert np.array_equal(features, features_again)
    assert np.array_equal(target, target_again)


def test_all_six_predictions_are_deterministic_and_positive():
    first = train_and_predict()
    second = train_and_predict()
    assert JOB_FEATURES.shape == (6, 3)
    assert first.predictions.shape == (6,)
    assert np.all(first.predictions > 0)
    assert np.array_equal(first.predictions, second.predictions)
    assert np.isfinite(first.mae) and np.isfinite(first.r2)

