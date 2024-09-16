import numpy as np
import denoiset.model as model


def test_generate_model_3d():
    """ Check that setting random seed yields the
    same initalization. """
    seed_value = np.random.randint(10)
    model_a = model.generate_model_3d(seed_value)
    model_b = model.generate_model_3d(seed_value)
    model_c = model.generate_model_3d()

    assert str(model_a.state_dict()) == str(model_b.state_dict())
    assert str(model_a.state_dict()) != str(model_c.state_dict())
