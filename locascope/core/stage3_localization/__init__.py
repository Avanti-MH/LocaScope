'''Stage 3 -- localise inside a candidate. A real package: import
its modules fully qualified, `from stage3_localization.SIFT_RANSAC import
SiftRansacLocalizer`. The directory is not on sys.path; see locascope/paths.py.
'''

from locascope.core.config.ConfigIdentity import method_recipe

#: The stage-3 methods: method -> (module, its recipe table, the localizer).
#: `--stage3 sift:default`.
METHODS = {
    'sift': ('locascope.core.stage3_localization.SIFT_RANSAC', 'SIFT_RECIPES',
             'SiftRansacLocalizer'),
}


def recipe(spec: str):
    """`(method, recipe, config, localizer class)` for `<method>:<recipe>`."""
    return method_recipe(spec, METHODS)
