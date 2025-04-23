# Global exports for backward compatibility
from .pretraining.data_loader import (
    load_acf_data_unsupervised,
    load_csi_data_unsupervised,
    load_data_unsupervised,
    load_preprocessed_data_unsupervised
)

from .pretraining.model_loader import (
    load_model_unsupervised,
    load_model_unsupervised_joint,
    load_model_unsupervised_joint_csi_var,
    load_model_unsupervised_joint_fix_length
)

from .supervised.data_loader import (
    load_data_supervised,
    save_data_supervised,
    load_acf_supervised,
    load_acf_unseen_environ,
    load_acf_supervised_NTUHumanID,
    load_acf_unseen_environ_NTU,
    load_acf_supervised_NTUHumanID_fewshot,
    load_csi_supervised_integrated,
    load_csi_unseen_integrated
)

from .supervised.model_loader import (
    load_model_pretrained,
    fine_tune_model,
    load_model_trained,
    load_model_scratch
)

from .meta_learning.data_loader import (
    load_csi_data_benchmark
)

from .meta_learning.model_loader import (
    load_csi_model_benchmark
)

# Factory functions for easier API access
from .base import (
    get_data_loader,
    get_model_loader
)