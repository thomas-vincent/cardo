import tree
import logging

TABLE_OPT_WIDTH_TO_HEIGHT_RATIO = 1 / (2**.5)

logger = logging.getLogger('cardo')

def make_table_from_folder(data_path, data_file_pattern, max_depth=-1,
                           level_names=None, row_levels=None,
                           column_levels=None):
    """
    Organize image files found in given folder hierarchy into a n-dimensional
    table and return it as SVG.

    Args:
        - data_path (str):
            Root of the folder hierarchy of images organized as a UB-tree:
            Example:
                 `-- root
                    |-- level1_val11
                    |   |-- level2_val21
                    |   |   `-- data_t1.png
                    |   |   `-- data_t2.png
                    |   |-- level2_val22
                    |   |   `-- data_t1.png
                    |   |   `-- data_t2.png
                    |   `-- ...
                    `-- level1_val12
                        |-- level2_val21   
                        |   `-- data_t1.png  
                        |   `-- data_t2.png
                        |-- level2_val22   
                        |   `-- data_t1.png  
                        |   `-- data_t2.png
                        `-- ...
             which represents a tree with 3 levels:
                  - lvl1_val1, lvl1_val2 
                  - lvl2_val1, lvl2_val2
                  - data_t1, data_t2
        - data_file_pattern (str | _sre.SRE_Pattern):
           Regular expression (see re package) to select data files
           when hierarchy end or max_depth is reached
        - max_depth (int):
           Maximum number of levels to parse in the folder hierarchy.
           If -1: go til the end.
        - level_names (list of str):
           names of levels in the folder hierarchy
        - row_levels (list of str):
           level names to show as rows in the final table. 
           If row_levels is provided then level_names must be given too.
        - column_levels (list of str):
           level names to show as colums in the final table.
           If column_levels is provided then level_names must be given too.

    Output (str):
        SVG string reprensentation of the table

    #TODO: TEST level_names, row_levels, column_levels args
    #TODO: check consistency between given branch names and
    #      retrieved ones
    """
    level_names =  level_names or []

    # Walk given folder and build data tree
    dtree, dfiles_bnames = tree.dtree_from_folder(data_path, data_file_pattern,
                                                  max_depth)

    dtree_depth = tree.dtree_get_depth(dtree)
    if len(level_names) == 0:
        branch_prefix = 'tmpbn'
        if dfiles_bnames is not None:
            for dfbn in dfiles_bnames:
                assert branch_prefix not in dfbn
        level_names = [branch_prefix + str(i) \
                        for i in range(dtree_depth-len(dfiles_bnames))]
    
    level_names.extend(dfiles_bnames)
        
    assert len(level_names) == dtree_depth

    if row_levels is None:
        if column_levels is None:
            row_levels, column_levels = opt_row_col_levels(dtree, level_names)
        else:
            assert set(column_levels).issubset(level_names)
            row_levels = list(set(level_names).difference(column_levels))

    if column_levels is None:
        assert set(row_levels).issubset(level_names)
        column_levels = list(set(level_names).difference(row_levels))
                
    # Forge final table with target rows and columns into a SVG document 
    return tree.dtree_to_svg(dtree, data_path, level_names, row_levels,
                             column_levels)

def opt_row_col_levels(dtree, level_names=None):
    """
    Optimize table layout so that it's more likely
    to fit on an A4 sheet
    """
    opt_ratio = TABLE_OPT_WIDTH_TO_HEIGHT_RATIO

    # get layer sizes
    layer_sizes = [len(l)*1. for l in tree.dtree_get_levels(dtree)]

    if level_names is None:
        level_names = range(len(layer_sizes))
    
    # find combination of row and column that is closest to opt_ratio

    cur_ratio = 10. #starting dummy value
    for row_layers, col_layers in bipartition_it(range(len(layer_sizes))):
        ratio = sum(layer_sizes[i] for i in col_layers) / \
                max(sum(layer_sizes[i] for i in row_layers), 1e-6)
        if abs(ratio - opt_ratio) < abs(cur_ratio - opt_ratio):
            cur_ratio = ratio
            opt_rows = row_layers
            opt_cols = col_layers

    return [level_names[i] for i in opt_rows], \
        [level_names[i] for i in opt_cols]

def bipartition_it(elements):
    def _bpart_rec(elems, set1, set2):
        if len(set1) + len(set2) < len(elements):
            for s1,s2 in _bpart_rec(elems[1:], set1 + [elems[0]], set2):
                yield s1, s2
            for s1,s2 in _bpart_rec(elems[1:], set1, set2 + [elems[0]]):
                yield s1, s2
        else:
            yield set1, set2
    return _bpart_rec(elements, [], [])
