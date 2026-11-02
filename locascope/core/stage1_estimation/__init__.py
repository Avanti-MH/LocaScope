'''Stage 1 -- estimate the query's mpp. A real package: import
its modules fully qualified, `from stage1_estimation.KnnEstMpp import
KnnEstMpp`. The directory is not on sys.path; see locascope/paths.py.
'''

from locascope.core.config.ConfigIdentity import method_recipe

#: The stage-1 methods: method -> (module, its recipe table, the estimator).
#: `--stage1 knn:gigapath`.
METHODS = {
    'knn':        ('locascope.core.stage1_estimation.KnnEstMpp', 'KNN_RECIPES', 'KnnEstMpp'),
    'classifier': ('locascope.core.stage1_estimation.ClassifierEstMpp', 'CLASSIFIER_RECIPES',
                   'ClassifierEstMpp'),
    'prototype':  ('locascope.core.stage1_estimation.PrototypeEstMpp', 'PROTOTYPE_RECIPES',
                   'PrototypeEstMpp'),
    'classic':    ('locascope.core.stage1_estimation.estimate_mpp_classic', 'CLASSIC_RECIPES',
                   'ClassicEstMpp'),
}


def recipe(spec: str):
    """`(method, recipe, config, estimator class)` for `<method>:<recipe>`."""
    return method_recipe(spec, METHODS)
