import os
import os.path as op
import re
from pprint import pformat
from itertools import izip, cycle
import logging

import numpy as np
import svgwrite

from cardo import graphics

logger = logging.getLogger('cardo')

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

            # print 'tfiles:', pformat(tfiles)
            # print 'bfiles:', pformat(bfiles)
            # print 'actual tree:'
            # print pformat(tree)
            # print ''

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
                    tfiles = apply_to_leaves(tfiles, make_path, root)
                    for bf,fn in tree_items_iterator(tfiles):
                        set_tree_leaf(tree, branches+bf, fn)
                
            dirs[:] = [] #do not go deeper
            
    return dtree_check(tree), bfiles


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

# def dtree_add_spacers(dtree, n_lvls_rows, n_lvls_cols, row_bgap, col_bgap):
#     def _add_spacers_rec(dt, ilvl):

#         if isinstance(dt, dict):
#             for subt in dt.itervalues():
#                 _add_spacers_rec(subt, ilvl+1)

#             if ilvl < n_lvls_rows:
#                 sgap = ' ' * row_bgap * (n_lvls_rows - 1 - ilvl)
#             else:
#                 sgap = ' ' * col_bgap * (n_lvls_rows + n_lvls_cols - 1 - ilvl)
#             if len(sgap) > 0:
#                 dt[sgap] = None
                
#     _add_spacers_rec(dtree, 0)

def make_headers(row_levels, col_levels):
    # Build boxed elements for headers:
    def mk_hdr_btexts(levels):        
        hdr_btexts = []
        cum_prod = 1
        for lvl in levels:
            cum_prod *= len(lvl)
            clvl = cycle(sorted(lvl))
            btexts = []
            for i in xrange(cum_prod):
                logger.debug('i=%d', i)
                txt = clvl.next()
                btexts.append(graphics.BoxedText(txt))
            hdr_btexts.append(btexts)
        return hdr_btexts
    logger.debug('Make text elements for row header ...')
    row_hdr_btexts = mk_hdr_btexts(row_levels)
    logger.debug('Make text elements for column header ...')
    col_hdr_btexts = mk_hdr_btexts(col_levels)

    return row_hdr_btexts, col_hdr_btexts

def space_headers(row_hdr_btexts, row_levels, row_base_gap,
                  col_hdr_btexts, col_levels, col_base_gap):
    txt_h = graphics.BoxedText.DEFAULT_FONT_H
    def _space_header(hdr, lvls, gap, ):
        rlvls = list(reversed(lvls))
        relems = reversed(hdr)
        spaced_elems = []
        for ilvl, (lvl, hrd_elems) in enumerate(zip(rlvls, relems)):
            spaced_line = []
            for ie, elem in enumerate(hrd_elems):
                # Add some before the current element
                if ie>0: # only interleaved gaps
                    if ie%len(lvl) != 0: #gap only concern current line
                        spaced_line.append(graphics.Spacer(width=(ilvl) * gap,
                                                           height=txt_h))
                    else: #gap depends on parent level(s)
                        ref_level = lvl
                        tmp_ie = ie
                        nspacers = ilvl
                        for lvl_parent in rlvls[ilvl+1:]:
                            if tmp_ie%len(ref_level) == 0:
                                nspacers += 1
                            else:
                                break
                            tmp_ie /= len(ref_level)
                            ref_level = lvl_parent
                        spaced_line.append(graphics.Spacer(nspacers * gap,
                                                           height=txt_h))
                # append current element
                spaced_line.append(elem)
            spaced_elems.insert(0, spaced_line) #take into account reversed order
        return spaced_elems

    return (_space_header(hdr, lvls, gap) \
            for hdr, lvls, gap in ((row_hdr_btexts, row_levels, row_base_gap),
                                   (col_hdr_btexts, col_levels, col_base_gap)))

def dtree_to_svg(dtree, root_path, branch_names, row_branches, column_branches,
                 row_base_gap=2, col_base_gap=2):
    # Reshape tree to match target row and column axes 
    dtree = tree_rearrange(dtree, branch_names, row_branches + column_branches)
    levels = dtree_get_levels(dtree)

    row_levels = levels[:len(row_branches)]
    col_levels = levels[len(row_branches):]
    row_hdr_btexts, col_hdr_btexts = make_headers(row_levels, col_levels)
    row_hdr_btexts, col_hdr_btexts = space_headers(row_hdr_btexts, row_levels,
                                                   row_base_gap,
                                                   col_hdr_btexts, col_levels,
                                                   col_base_gap)
    # Set image size relative to text size:
    img_h = graphics.BoxedText.DEFAULT_FONT_H * 7    
    img_it = tree_leaves_iterator(dtree)
    all_bimgs = np.empty((len(row_hdr_btexts[-1]), len(col_hdr_btexts[-1])),
                         dtype=object)
    for irow, elem in enumerate(row_hdr_btexts[-1]):
        if isinstance(elem, graphics.Spacer): # spacer
            # Add a full row of spacers
            sp_h = elem.get_box_width()
            for icol in xrange(len(col_hdr_btexts[-1])):
                all_bimgs[irow, icol] = graphics.Spacer(height=sp_h)
        else:
            # interleave spacers between columns
            for icol, elem in enumerate(col_hdr_btexts[-1]):
                if isinstance(elem, graphics.Spacer): # spacer
                    ew = elem.get_box_width()
                    all_bimgs[irow, icol] = graphics.Spacer(width=ew)
                else:
                    bimg = graphics.BoxedImage(op.join(root_path, img_it.next()),
                                               img_h=img_h)
                    all_bimgs[irow, icol] = bimg
            
    bimgs_array = np.array(all_bimgs)
    bimgs_array = bimgs_array.reshape(len(row_hdr_btexts[-1]),
                                      len(col_hdr_btexts[-1]))
    # print 'bimgs_array.shape:', bimgs_array.shape
    
    # Adjust sizes of elements to avoid truncation 
    graphics.adjust_table_sizes(row_hdr_btexts, col_hdr_btexts, bimgs_array)
    # Position all elements 
    graphics.arrange_table(row_hdr_btexts, col_hdr_btexts, bimgs_array)

    # Flatten every part of the table (row hdr, col hdr and img grid)
    # -> will be easer to put into SVG groups


    # Put everything in a svg document
    # Apply 90 deg. rotation + translation to the column header    
    # Apply global translation to put elements in their right position
    # relative to top-left corner (which is point (0,0))
    
    # svg_doc = create_empty_svg_doc()
    # col_hdr_w, col_hdr_h = add_column_headers(svg_doc, column_branches,
    #                                           col_levels)
    # row_hdr_w, row_hrd_h = add_row_headers(svg_doc, row_branches, row_levels)

    dwg = svgwrite.Drawing()
    table_group = svgwrite.container.Group(id='table')
    
    col_hdr_group = svgwrite.container.Group(id='table_col_hdr')
    for col_hdr_line in col_hdr_btexts:
        for element in col_hdr_line:
            col_hdr_group.add(element.to_svg())            
    table_group.add(col_hdr_group)

    row_hdr_group = svgwrite.container.Group(id='table_row_hdr')
    for row_hdr_line in row_hdr_btexts:
        for element in row_hdr_line:
            row_hdr_group.add(element.to_svg())

    # translate row hdr so that its bottom left corner is
    # on the bottom left corner of the image grid
    img_bot_y = row_hdr_btexts[-1][0].get_box_bot_y()
    dy = bimgs_array[-1,0].get_box_bot_y() - img_bot_y
    row_hdr_group.translate(tx=0,ty=dy)

    # rotate row hdr by 90 deg. around the bottom left corner of the image grid
    rot_ctr = (0, img_bot_y)
    row_hdr_group.rotate(-90, center=rot_ctr)    
    table_group.add(row_hdr_group)
    
    img_group = svgwrite.container.Group(id='table_content')
    for row_imgs in bimgs_array:
        for img in row_imgs:
            img_group.add(img.to_svg())
    table_group.add(img_group)

    dwg.add(table_group)
    return dwg.tostring()


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



                    
