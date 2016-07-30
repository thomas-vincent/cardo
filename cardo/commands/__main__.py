import re
import logging
from optparse import OptionParser

import cardo

logger = logging.getLogger('cardo')

def main():
    #TODO add command unit test
    
    min_args = 2
    max_args = 2
    
    usage = 'usage: %prog [options] PATH FILE_REGEXP'
    description = 'Parse given hierarchy PATH, find images captured '\
                  'within each subfolder by FILE_REGEXP and organize them '\
                  'as a table in a SVG document. Consider using quotes to '\
                  'protect FILE_REGEXP.' 
    
    parser = OptionParser(usage=usage, description=description)
    
    parser.add_option('-v', '--verbose', dest='verbose', metavar='VERBOSELEVEL',
                      type='int', default=0,
                      help='Amount of verbose: '\
                           '0 (NOTSET: quiet, default), '\
                           '50 (CRITICAL), ' \
                           '40 (ERROR), ' \
                           '30 (WARNING), '\
                           '20 (INFO), '\
                           '10 (DEBUG)')
    
    def check_fraction(value):
        return value >= 0 and value <= 1
    
    def strsplit(option, opt_str, value, parser):
        setattr(parser.values, option.dest, value.split(','))

    parser.add_option('-l', '--sublevel-names', metavar='LIST_OF_NAMES',
                      type='string', action='callback', callback=strsplit, 
                      help='define names of all subfolder levels to parse. '
                           'By default, the name of one level is its depth')

    
    parser.add_option('-c', '--columns', metavar='LIST_OF_NAMES', type='string',
                      action='callback', callback=strsplit,
                      help='Names of subfolder level to put as columns in the '\
                           'final table')

    parser.add_option('-r', '--rows', metavar='LIST_OF_NAMES', type='string',
                      action='callback', callback=strsplit,
                      help='Names of subfolder level to put as rows in the '\
                           'final table')

    #TODO TEST gaps
    parser.add_option('-g', '--column-gap-increment', metavar='SIZE_PX',
                      type='int', default=cardo.graphics.Table.DEFAULT_COL_BGAP,
                      help='Set the column spacing between levels in px.' \
                           'Gap is increased by this increment from the '\
                           'bottom to the top level, starting from 0.')

    parser.add_option('-a', '--row-gap-increment', metavar='SIZE_PX',
                      type='int', default=cardo.graphics.Table.DEFAULT_ROW_BGAP,
                      help='Set the row spacing between levels in px.' \
                           'Gap is increased by this increment from the '\
                           'bottom to the top level, starting from 0.')
    
    parser.add_option('-o', '--output-file', metavar='FILE', type='string',
                      help='Output SVG file where to save table. Default: '\
                           'print SVG content to stdout')
    
    (options, args) = parser.parse_args()
    logger.setLevel(options.verbose)
    
    nba = len(args)
    if nba < min_args or (max_args >= 0 and nba > max_args):
        parser.print_help()
        return 1
    
    data_path, fn_reg_exp = args

    fn_reg_exp = re.compile(fn_reg_exp)

    if not check_fraction(options.column_gap_increment):
        print 'Wront value for option --column-gap-increment'
        parser.print_help()
        return 1

    if not check_fraction(options.row_gap_increment):
        print 'Wront value for option --row-gap-increment'
        parser.print_help()
        return 1
    
    svg_table = cardo.make_table_from_folder(data_path, fn_reg_exp,
                                             level_names=options.sublevel_names,
                                             row_levels=options.rows,
                                             column_levels=options.columns)

    if options.output_file is not None:
        # Save SVG document
        with open(options.output_file, 'w') as fout:
            fout.write(svg_table)
    else:
        print svg_table
