'''Stage 1 -- estimate the query's mpp. A real package: import
its modules fully qualified, `from stage1_estimation.KnnEstMpp import
KnnEstMpp`. The directory is not on sys.path; see utilities/_paths.py.
'''

from ConfigIdentity import method_recipe

#: The stage-1 methods: method -> (module, its recipe table, the estimator).
#: `--stage1 knn:gigapath`.
METHODS = {
    'knn':        ('stage1_estimation.KnnEstMpp', 'KNN_RECIPES', 'KnnEstMpp'),
    'classifier': ('stage1_estimation.ClassifierEstMpp', 'CLASSIFIER_RECIPES',
                   'ClassifierEstMpp'),
    'prototype':  ('stage1_estimation.PrototypeEstMpp', 'PROTOTYPE_RECIPES',
                   'PrototypeEstMpp'),
    'classic':    ('stage1_estimation.estimate_mpp_classic', 'CLASSIC_RECIPES',
                   'ClassicEstMpp'),
}


def recipe(spec: str):
    """`(method, recipe, config, estimator class)` for `<method>:<recipe>`."""
    return method_recipe(spec, METHODS)
