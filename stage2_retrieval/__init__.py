'''Stage 2 -- retrieve candidate windows. A real package: import
its modules fully qualified, `from stage2_retrieval.SlidingWinSimRot
import ...`. The directory is not on sys.path; see utilities/_paths.py.
'''

from ConfigIdentity import method_recipe

#: The stage-2 methods: method -> (module, its recipe table, the retriever).
#: `--stage2 slidewin:gigapath`.
METHODS = {
    'slidewin': ('stage2_retrieval.SlidingWinSimRot', 'SLIDEWIN_RECIPES',
                 'SlidingWinSimRot'),
}


def recipe(spec: str):
    """`(method, recipe, config, retriever class)` for `<method>:<recipe>`."""
    return method_recipe(spec, METHODS)
