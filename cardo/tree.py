"""
Provide operations on simple trees defined as nested python dicts:
- get / setter of leaves
- iteration over branches, leaves and items (both branches and leaves)
- level rearranging 

Provide operations on UB-trees (called DataTree here):
- build from hierarchical folders of images
- check structure
"""
import os
import os.path as op
import re
from pprint import pformat
from itertools import izip
import logging

logger = logging.getLogger('cardo')

## Basic tree functions ##

def get_tree_leaf(element, branch):
    """
    Return the nested leaf element corresponding to all dictionnary keys in
    branch from element
    """
    if isinstance(element, dict) and len(branch) > 0:
        return get_tree_leaf(element[branch[0]], branch[1:])
    else:
        return element


def is_tree_leaf(tree, branch, leaf):
    if isinstance(tree, dict) and len(branch) > 0 and tree.has_key(branch[0]):
        return is_tree_leaf(tree[branch[0]], branch[1:], leaf)
    elif len(branch) == 0:
        return tree == leaf
    else:
        return False
    
    
def set_tree_leaf(tree, branch, leaf, branch_classes=None):
    """
    Set the nested *leaf* element corresponding to all dictionnary keys
    defined in *branch*, within *tree*
    """
    assert isinstance(tree, dict)
    if len(branch) == 1:
        tree[branch[0]] = leaf
        return
    if not tree.has_key(branch[0]):
        if branch_classes is None:
            tree[branch[0]] = tree.__class__()
        else:
            tree[branch[0]] = branch_classes[0]()
    else:
        assert isinstance(tree[branch[0]], dict)
    if branch_classes is not None:
        set_tree_leaf(tree[branch[0]], branch[1:], leaf, branch_classes[1:])
    else:
        set_tree_leaf(tree[branch[0]], branch[1:], leaf)


def swap_layers(t, labels, l1, l2):
    """ Create a new tree from t where layers labeled by l1 and l2 are swapped.
    labels contains the branch labels of t.
    """
    i1 = labels.index(l1)
    i2 = labels.index(l2)
    nt = t.__class__()  # new tree init from the input tree
    # can be dict, OrderedDict, SortedDict or ...
    for b, l in izip(tree_branches_iterator(t), tree_leaves_iterator(t)):
        nb = list(b)
        nb[i1], nb[i2] = nb[i2], nb[i1]
        set_tree_leaf(nt, nb, l)

    return nt


def tree_rearrange(t, branches, new_branches):
    """ Create a new tree from t where layers are rearranged following 
    new_branches.
    Arg branches contains the current branch labels of t.
    """
    order = [branches.index(nl) for nl in new_branches]
    nt = t.__class__()  # new tree
    for b, l in izip(tree_branches_iterator(t), tree_leaves_iterator(t)):
        nb = [b[i] for i in order]
        set_tree_leaf(nt, nb, l)

    return nt

def tree_leaves_iterator(tree):
    for branch in tree_branches_iterator(tree):
        yield get_tree_leaf(tree, branch)


def tree_branches_iterator(tree, branch=None):
    if branch is None:
        branch = []
    if isinstance(tree, dict):
        for k in tree.iterkeys():
            for b in tree_branches_iterator(tree[k], branch + [k]):
                yield b
    else:
        yield branch


def tree_items_iterator(tree):
    """
    """
    for branch in tree_branches_iterator(tree):
        yield (branch, get_tree_leaf(tree, branch))


def apply_to_leaves(tree, func, func_args=None, func_kwargs=None):
    """
    Apply function 'func' to all leaves in given 'tree' and return a new tree.
    """
    if func_kwargs is None:
        func_kwargs = {}
    if func_args is None:
        func_args = []

    newTree = tree.__class__()  # could be dict or {}
    for branch, leaf in tree_items_iterator(tree):
        set_tree_leaf(newTree, branch, func(leaf, *func_args, **func_kwargs))
    return newTree

        
## Data tree (= UB-tree) functions ##        

class InconsistentFileGroup(Exception):
    pass

class NonMatchingFilePattern(Exception):
    pass

def file_list_to_tree(files, file_pat):

    if file_pat.groups > 0: # there are groups in the regexp.
                            # File list may actually be represented as
                            # a data tree
        branch_names = [None] * file_pat.groups
        for gname, gidx in file_pat.groupindex.iteritems():
            branch_names[gidx-1] = gname
        branch_names = [bn for bn in branch_names if bn is not None]
            
        ftree = {}
        for fn in sorted(files):
            m = file_pat.match(fn)
            if m is not None:
                set_tree_leaf(ftree, m.groups(), fn)
        return ftree, branch_names
    else: # no groups in the regexp
          # regexp should only capture one file #TODO: document this
        selected_file = None
        for fn in files:
            m = file_pat.match(fn)
            if m is not None:
                if selected_file is None:
                    selected_file = fn
                else:
                    raise Exception('Regexp "%s" has no group but catches '\
                                    'more than one single file' \
                                    %file_pat.pattern)
            if selected_file is None:
                raise Exception('Regexp "%s" did not catch any file' \
                                %file_pat.pattern)
        return selected_file, []

def dtree_from_folder(startpath, file_pattern, max_depth=-1):
    """
    Generate a data tree from the given folder hierarchy.
    An edge correspond to a subfolder and a leaf to a file whose basename matches
    the given file_pattern. Leaf files are relative to given startpath.
    TODO: handle case where recursion stops at 1st level and there is only one 
          single file to handle with no hierarchy. 
          In this case, dtree cannot be built.
    """
    tree = {}

    try:
        file_pattern.match('')
    except AttributeError:
        file_pattern = re.compile(file_pattern)

    logger.info('Parse folder: %s', startpath)
    logger.info('File regexp: %s', file_pattern.pattern)

    for root, dirs, files in os.walk(startpath, topdown=True):
        logger.info('Parse subfolder: %s', root)
        dirs.sort()
        rel_path_parts = op.relpath(root, startpath).split(op.sep)
        depth = len(rel_path_parts)

        if root == startpath:
            branches = []
        else:
            branches = rel_path_parts
            
        if (max_depth !=- 1 and depth == max_depth) or \
           len(dirs) == 0:
            logger.info('Maximum depth reached')
                        
            # No more subfolders available or given max depth is reached
            # Build leaves from current files
            
            # TODO: clarify return value of file_list_to_tree
            #       for now it's either a single file or a dtree
            #       -> better to unify
            tfiles, bfiles = file_list_to_tree(files, file_pattern)

            if len(branches) == 0 and len(bfiles) == 0:
                raise Exception('Data tree cannot be built from one single file')

            def make_path(fn, rdir):
                return op.join(op.relpath(rdir, startpath), fn)

            if isinstance(tfiles, str):
                logger.info('Found 1 file matching regexp: %s', tfiles)
                tfiles = make_path(tfiles, root)
                set_tree_leaf(tree, branches, tfiles)
            else: # dict tree
                if logger.isEnabledFor(logging.INFO):
                    logger.info('Files matching regexp:\n%s',
                                pformat(list(tree_leaves_iterator(tfiles))))
                try:
                    dtree_check(tfiles)
                except WrongDataTreeLevel as ewt:
                    #TODO: improve error handling and distinguish between
                    #      missing file or wrong group def
                    msg = 'Wrong number of image files across subfolders or '\
                          'ambiguous group definitions in given '\
                          'file pattern, consider using (?:) to '\
                          'define non-capturing groups.\n'\
                          'First items of wrong group are:\n%s' \
                          %','.join(ewt.ref_lvl+ewt.wrong_lvl)
                    raise InconsistentFileGroup(msg)
                except EmptyDataTree:
                    msg = 'No file matched pattern:\n %s in folder:\n %s' \
                          %(file_pattern.pattern, root)
                    logger.warning(msg)
                    #raise NonMatchingFilePattern(msg)
                else:
                    tfiles = apply_to_leaves(tfiles, make_path, (root,))
                    for bf,fn in tree_items_iterator(tfiles):
                        set_tree_leaf(tree, branches+bf, fn)
                
            dirs[:] = [] #do not go deeper
            
    return dtree_check(tree), bfiles

class WrongDataTreeLevel(Exception):

    def __init__(self, message, wrong_level_idx, ref_lvl, wrong_lvl):
        super(WrongDataTreeLevel, self).__init__(message)
        self.wrong_level_idx = wrong_level_idx
        self.ref_lvl = ref_lvl
        self.wrong_lvl = wrong_lvl
            
class WrongDataTreeLeaf(Exception):
    pass

class EmptyDataTree(Exception):
    pass


def dtree_get_depth(data_tree):
    def _count_levels(t, count):
        if isinstance(t[t.keys()[0]], dict):
            return _count_levels(t[t.keys()[0]], count+1)
        else:
            return count + 1
    return _count_levels(data_tree, 0)

def dtree_get_levels(data_tree):
    # Walk one branch of the tree and gather levels
    def _get_levels(t, levels):
        if isinstance(t[t.keys()[0]], dict):
            return _get_levels(t[t.keys()[0]], levels + [t.keys()])
        else:
            return levels + [t.keys()]
    return _get_levels(data_tree, [])
    
def dtree_check(data_tree):
    """
    Ensure that given tree is a data tree: all non-leaf nodes at any 
    level have the same edges. All leaves are of the same type.
    """
    if len(data_tree) == 0:
        raise EmptyDataTree('Empty data tree')
    
    ref_levels = [set(l) for l in dtree_get_levels(data_tree)]
    
    # Ensure that other levels are the same as those of the 1st branch
    ref_leaf_type = tree_leaves_iterator(data_tree).next().__class__
    def _check_levels(t, levels, cur_level):
        if len(levels) > 0:
            if len(levels) == 0: # leaf level reached
                # only check for consistent leaf type
                if t.itervalues().next().__class__ != ref_leaf_type:
                    raise WrongDataTreeLeaf('Inconsistent data tree: '\
                                            'leaf type not homogeneous')
            else: # non-leaf level
                # check for consistent level
                if set(t.keys()) != levels[0]:
                    msg = 'Inconsistent data tree. Error at level %d.\n' \
                          '- ref level:\n %s\n- wrong level:\n %s\n' \
                          %(cur_level, pformat(levels[0]),pformat(t.keys()))
                    raise WrongDataTreeLevel(msg, cur_level, list(levels[0]),
                                             t.keys())
                                          
                for subt in t.itervalues():
                    _check_levels(subt, levels[1:], cur_level+1)
    _check_levels(data_tree, ref_levels, 0)    
    return data_tree

def dtree_to_svg(dtree, root_path, branch_names, row_branches, column_branches,
                 row_base_gap=None, col_base_gap=None):
    raise DeprecationWarning('cardo.tree.dtree_to_svg has been moved to '\
                             'graphics.dtree_to_svg')
