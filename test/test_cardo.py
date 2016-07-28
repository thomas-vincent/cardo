import unittest
import tempfile
import shutil
import os.path as op
import os
from pprint import pformat
import re
import numpy as np
import base64
import math
import logging

from .context import cardo
gfx = cardo.graphics

logger = logging.getLogger('cardo')

#cd ..;python -m unittest test.test_cardo

class CardoTest(unittest.TestCase):

    def setUp(self):
        if 1:
            self.tmp_dir = tempfile.mkdtemp(prefix='cardo_tmp_')
        else:
            self.tmp_dir = '/tmp/cardo_test'
            if not op.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir)
        self.clean_tmp = True
        
    def tearDown(self):
        if self.clean_tmp:
            shutil.rmtree(self.tmp_dir)

    def _create_empty_tmp_files(self, fns):
        """
        Create empty files from given filenames in the
        current temporary directory (see self.tmp_dir)
        """
        tmp_fns = [op.join(self.tmp_dir,fn) for fn in fns]
        for fn in tmp_fns:
            d = op.dirname(fn)
            if not op.exists(d):
                os.makedirs(d)
            open(fn, 'a').close()
        return tmp_fns
            
    def _create_img_files(self, fns):
        img_content = get_growth_profile_img_content()
        tmp_fns = [op.join(self.tmp_dir,fn) for fn in fns]
        for fn in tmp_fns:
            d = op.dirname(fn)
            if not op.exists(d):
                os.makedirs(d)
                
            with open(fn, 'a') as fimg:
                fimg.write(img_content)

        return tmp_fns
                
    def assert_file_exists(self, fn):
        if not op.exists(fn):
            raise Exception('File %s does not exist' %fn)


    def assert_lists_are_equal(self, l1, l2):
        if len(l1) != len(l2):
            raise Exception('List differ (len#):\nl1: ' + \
                            str(l1) + '\nl2: ' + str(l2))
        for e1,e2 in zip(l1,l2):
            if e1 != e2:
                raise Exception('List differ (#elems):\nl1: ' + \
                                str(l1) + '\nl2: ' + str(l2))
                
class MainTest(CardoTest):
    
    def test_make_table_from_folder_simple_leaf(self):
        
        tmp_files = ['my_study/scenario1/experiment1/growth_profile.png',
                     'my_study/scenario1/experiment2/growth_profile.png',
                     'my_study/scenario1/experiment3/growth_profile.png',
                     'my_study/scenario2/experiment1/growth_profile.png',
                     'my_study/scenario2/experiment2/growth_profile.png',
                     'my_study/scenario2/experiment3/growth_profile.png']
        
        self._create_img_files(tmp_files)
        
        out_file = op.join(self.tmp_dir, 'growth_table.svg')
        svg = cardo.make_table_from_folder(op.join(self.tmp_dir, 'my_study'),
                                           'growth_profile.png')
        
        if 0:
            # Save SVG document and manually check with inkscape
            with open(out_file, 'w') as f:
                f.write(svg)

        #TODO: check some svg content
        
    def test_make_table_from_folder_cplx_leaves(self):
        
        tmp_files = ['my_study/scenario1/experiment1/side_right_stim_1.png',
                     'my_study/scenario1/experiment1/side_left_stim_1.png',
                     'my_study/scenario1/experiment1/side_left_stim_3.png',
                     'my_study/scenario1/experiment1/side_right_stim_3.png',
                     'my_study/scenario1/experiment1/side_left_stim_2.png',
                     'my_study/scenario1/experiment1/side_right_stim_2.png',
                     'my_study/scenario1/experiment2/side_left_stim_1.png',
                     'my_study/scenario1/experiment2/side_right_stim_1.png',
                     'my_study/scenario1/experiment2/side_left_stim_2.png',
                     'my_study/scenario1/experiment2/side_right_stim_2.png',
                     'my_study/scenario1/experiment2/side_left_stim_3.png',
                     'my_study/scenario1/experiment2/side_right_stim_3.png']
        
        self._create_img_files(tmp_files)
        
        fpat = 'side_(?P<side>(?:left|right))_(?P<stim_name>stim_[0-9]).png'
        svg = cardo.make_table_from_folder(op.join(self.tmp_dir, 'my_study'),
                                           fpat)
        #TODO: check SVG
        
    def test_bipartition(self):
        expected_paritions = [ ([0,1], []), ([0], [1]),
                               ([1], [0]), ([], [0, 1]) ]
        all_partitions = list(cardo.bipartition_it([0,1]))
        for i, (set1, set2) in enumerate(all_partitions):
            if set(set1) != set(expected_paritions[i][0]):
                raise Exception('Wrong parition set, got\n:%s\nExpected:\n%s'\
                                %(pformat(all_partitions),
                                  pformat(expected_paritions)))
        
    def test_opt_row_col_levels(self):
        def create_dtree(layers, leaves):
            def _create_dtree_rec(layers):
                if len(layers) == 1:
                    return dict( zip(layers[0], leaves) )
                else:
                    return dict( (e, _create_dtree_rec(layers[1:])) \
                                 for e in layers[0] )
            return _create_dtree_rec(layers)

        dtree = create_dtree([range(7), range(5), range(2)], ['a','b'])
        row_levels, col_levels = cardo.opt_row_col_levels(dtree)
        assert set(row_levels) == set([0,2]) or \
            set(row_levels) == set([1,3])
        
class TreeTest(CardoTest):

    def assert_trees_are_equal(self, t1, t2):
        not_in_t2 = []
        for (b1,l1) in cardo.tree.tree_items_iterator(t1):
            if not cardo.tree.is_tree_leaf(t2, b1, l1):
                not_in_t2.append((b1, l1))

        not_in_t1 = []
        for (b2,l2) in cardo.tree.tree_items_iterator(t2):
            if not cardo.tree.is_tree_leaf(t1, b2, l2):
                not_in_t1.append((b2, l2))

        if len(not_in_t1) > 0 or len(not_in_t2) > 0:
            def branch2str(b, l):
                return '/'.join(b) + ':' + str(l)
            raise Exception('trees differ:\n'\
                            '  - branches in t1 and not in t2:\n' + \
                            '\n'.join((('    '+branch2str(b,l) \
                                        for b,l in not_in_t2))) + '\n' + \
                            '  - branches in t2 and not in t1:\n' + \
                            '\n'.join((('    '+branch2str(b,l) \
                                        for b,l in not_in_t1))))
        
    def test_is_tree_leaf(self):

        t = {'a1':{'b1':1}, 'a2': 2 }
        self.assertTrue(cardo.tree.is_tree_leaf(t, ('a1','b1'), 1))
        self.assertTrue(cardo.tree.is_tree_leaf(t, ('a2',), 2))
        self.assertFalse(cardo.tree.is_tree_leaf(t, ('a2','b1'), 1))
        self.assertFalse(cardo.tree.is_tree_leaf(t, ('a1','b1', 'c1'), 2))
        self.assertFalse(cardo.tree.is_tree_leaf(t, ('a1',), 1))


    def test_dtree_depth(self):
        dtree = {'a1':{'b1':1, 'b2': 2}, 'a2':{'b1':3, 'b2': 4}}
        self.assertEquals(cardo.tree.dtree_get_depth(dtree), 2)

    def test_dtree_check(self):

        bad_dtree1 = {'a1':{'b1':1, 'b2': 2}, 'a2':{'b1':3, 'b4': 4}}
        self.assertRaises(cardo.tree.WrongDataTreeLevel,
                          cardo.tree.dtree_check, bad_dtree1)

        good_dtree = {'a1':{'b1':1, 'b2': 2}, 'a2':{'b1':3, 'b2': 4}}
        cardo.tree.dtree_check(good_dtree)
        

        bad_dtree2 = {'a1':{'b1':1, 'b2': 2}, 'a2':{'b1':3}}
        self.assertRaises(cardo.tree.WrongDataTreeLevel,
                          cardo.tree.dtree_check, bad_dtree2)
        

    def test_file_list_to_tree(self):
        files = ['profile_speed_1.2_feed_normal.png',
                 'profile_speed_1.5_feed_normal.png',
                 'profile_speed_1.5_feed_half.png',
                 'profile_stats.png'
                 'totally_something_else']
        pat = re.compile('^profile_speed_(?P<speed>\d+\.\d+)_' \
                         'feed_(?P<feed>(?:half|normal)).png')
        ftree, fbranches = cardo.tree.file_list_to_tree(files, pat)
        expected_ftree = {'1.2':{'normal':'profile_speed_1.2_feed_normal.png'},
                          '1.5':{'normal':'profile_speed_1.5_feed_normal.png',
                                 'half':'profile_speed_1.5_feed_half.png' }}
        expected_branches = ['speed', 'feed']
        self.assert_lists_are_equal(fbranches, expected_branches)
        self.assert_trees_are_equal(ftree, expected_ftree)
        
    def test_dtree_from_folder(self):
        tmp_files = ['my_study/scenario1/experiment1/growth_profile.png',
                     'my_study/scenario1/experiment2/growth_profile.png',
                     'my_study/scenario1/experiment3/growth_profile.png',
                     'my_study/scenario2/experiment1/growth_profile.png',
                     'my_study/scenario2/experiment2/growth_profile.png',
                     'my_study/scenario2/experiment3/growth_profile.png']

        self._create_empty_tmp_files(tmp_files)

        data_folder = op.join(self.tmp_dir, 'my_study')
        data_tree, _ = cardo.tree.dtree_from_folder(data_folder,
                                                    'growth_profile.png')

        fpat = 'scenario%d/experiment%d/growth_profile.png'
        expected_tree = {'scenario1' : { 'experiment1' : fpat%(1,1),
                                          'experiment2' : fpat%(1,2),
                                          'experiment3' : fpat%(1,3),
                                          },
                          'scenario2' : { 'experiment1' : fpat%(2,1),
                                          'experiment2' : fpat%(2,2),
                                          'experiment3' : fpat%(2,3),
                                          },
                        }
                          
        self.assert_trees_are_equal(data_tree, expected_tree)


    def test_make_headers(self):
        row_hdr, col_hdr = cardo.tree.make_headers([['g1', 'g2', 'g3'],
                                                    ['a', 'b', 'c'],
                                                    ['h1', 'h2', 'h3']],
                                                   [['left', 'right']])
        self.assertEquals(row_hdr[0][1].text, 'g2')
        self.assertEquals(row_hdr[1][3].text, 'a')
        self.assertEquals(row_hdr[2][6].text, 'h1')
        self.assertEquals(col_hdr[0][0].text, 'left')
        
    def test_hdr_spacing(self):
        Btxt = cardo.graphics.BoxedText
        rbg = 20 #px
        cbg = 30 #px

        #logger.setLevel(logging.DEBUG)

        row_lvls = [['g1','g2','g3'], ['a','b','c'], ['h1','h2','h3']]
        row_hdr = [ [Btxt(t) for t in row_lvls[0]],
                     [Btxt(t) for t in row_lvls[1]*3],
                     [Btxt(t) for t in row_lvls[2]*3*3] ]
        
        col_lvls = [['left', 'right']]
        col_hdr = [ [Btxt(t) for t in col_lvls[0]] ]

        sp_row_h, sp_col_h = cardo.tree.space_headers(row_hdr, row_lvls, rbg,
                                                      col_hdr, col_lvls, cbg)

        self.assertEquals(sp_row_h[0][0].text, 'g1')
        self.assertIsInstance(sp_row_h[0][1], gfx.Spacer)
        self.assertEquals(sp_row_h[0][1].get_box_width(), rbg*2)
        self.assertEquals(sp_row_h[0][2].text, 'g2')
        self.assertIsInstance(sp_row_h[0][3], gfx.Spacer)
        self.assertEquals(sp_row_h[0][3].get_box_width(), rbg*2)

        self.assertEquals(sp_row_h[1][0].text, 'a')
        self.assertIsInstance(sp_row_h[1][1], gfx.Spacer)
        self.assertEquals(sp_row_h[1][1].get_box_width(), rbg)
        self.assertIsInstance(sp_row_h[1][5], gfx.Spacer)
        self.assertEquals(sp_row_h[1][5].get_box_width(), rbg*2)

        self.assertEquals(sp_row_h[2][0].text, 'h1')
        self.assertIsInstance(sp_row_h[2][1], gfx.Spacer)
        self.assertEquals(sp_row_h[2][1].get_box_width(), 0)
        self.assertEquals(sp_row_h[2][2].text, 'h2')
        self.assertIsInstance(sp_row_h[2][3], gfx.Spacer)
        self.assertEquals(sp_row_h[2][3].get_box_width(), 0)
        self.assertEquals(sp_row_h[2][4].text, 'h3')
        self.assertIsInstance(sp_row_h[2][5], gfx.Spacer)
        self.assertEquals(sp_row_h[2][5].get_box_width(), rbg)
        self.assertIsInstance(sp_row_h[2][17], gfx.Spacer)
        self.assertEquals(sp_row_h[2][17].get_box_width(), rbg*2)
        self.assertIsInstance(sp_row_h[2][35], gfx.Spacer)
        self.assertEquals(sp_row_h[2][35].get_box_width(), rbg*2)

        
class GraphicsTest(CardoTest):

    def test_img_size(self):
        img_fn = self._create_img_files(['test.png'])[0]
        self.assertTupleEqual(cardo.graphics.get_image_size(img_fn),
                              (211, 239))
    
    def test_set_box_size(self):
        txt = 'wazzzaaaa'
        btxt = cardo.graphics.BoxedText(txt, 0, 0, 5, 5 )

        dfh = cardo.graphics.BoxedText.DEFAULT_FONT_H
        dfw = cardo.graphics.BoxedText.DEFAULT_FONT_W
        self.assertEquals(btxt.get_box_width(), len(txt) * dfw)
        self.assertEquals(btxt.get_box_height(), dfh)

        btxt = cardo.graphics.BoxedText(txt, 0, 0, 500, 50)
        self.assertEquals(btxt.get_box_width(), 500)
        self.assertEquals(btxt.get_box_height(), 50)

        
    def test_text_coords(self):
        txt = 'wazzzaaaa'
        aln = cardo.graphics.BoxedText.CENTER_ALIGN
        btxt = cardo.graphics.BoxedText(txt, 0, 0, 500, 50,
                                        text_halign=aln)
        
        txt_w, txt_h = btxt.get_text_size()
        expected_coords = (500/2. - txt_w/2., 50/2. - txt_h/2.)
        self.assertTupleEqual(btxt.get_text_coords(), expected_coords)
        
    def test_hdeoverlap(self):
        txt = 'w'
        btxts = [cardo.graphics.BoxedText(txt, 0, 0, 50, 70),
                 cardo.graphics.BoxedText(txt, 0, 0, 40, 60)]

        cardo.graphics.BoxedText.hdeoverlap(btxts, 10)

        self.assertEquals(btxts[0].box_x, 0)
        self.assertEquals(btxts[1].box_x, 60)

    def test_vdeoverlap(self):
        txt = 'w'
        btxts = [cardo.graphics.BoxedText(txt, 0, 0, 50, 70),
                 cardo.graphics.BoxedText(txt, 0, 0, 40, 60)]

        cardo.graphics.BoxedText.vdeoverlap(btxts, 10)

        self.assertEquals(btxts[0].box_y, 0)
        self.assertEquals(btxts[1].box_y, 80)

        
    def test_to_svg(self):
        txt = 'wazzzaaaa'
        btxt = cardo.graphics.BoxedText(txt, 0, 0, 50, 50)

        txt_svg = btxt.to_svg()

        self.assertEquals(float(txt_svg.attribs['y']), float('5.0'))
        self.assertEquals(txt_svg.text, txt)

    def test_gaps(self):

        tmp_files = ['my_study/scenario1/experiment1/side_right_stim_1.png',
                     'my_study/scenario1/experiment1/side_left_stim_1.png',
                     'my_study/scenario1/experiment1/side_left_stim_3.png',
                     'my_study/scenario1/experiment1/side_right_stim_3.png',
                     'my_study/scenario1/experiment1/side_left_stim_2.png',
                     'my_study/scenario1/experiment1/side_right_stim_2.png',
                     'my_study/scenario1/experiment2/side_left_stim_1.png',
                     'my_study/scenario1/experiment2/side_right_stim_1.png',
                     'my_study/scenario1/experiment2/side_left_stim_2.png',
                     'my_study/scenario1/experiment2/side_right_stim_2.png',
                     'my_study/scenario1/experiment2/side_left_stim_3.png',
                     'my_study/scenario1/experiment2/side_right_stim_3.png']
        
        self._create_img_files(tmp_files)
        fpat = 'side_(?P<side>(?:left|right))_(?P<stim_name>stim_[0-9]).png'
        #logger.setLevel(logging.DEBUG)
        svg = cardo.make_table_from_folder(self.tmp_dir, fpat)
        
        #TODO: check SVG

    def test_adjust_hdr(self):
        lg_txt = 'the very very very very very very very very very long row hdr'
        Bt = gfx.BoxedText
        Sp = gfx.Spacer
        gap = 20
        row_hdr_btexts = [[Bt(lg_txt)],
                          [Bt('r1a'), Sp(gap), Bt('r1b')],
                          [Bt('r2a'), Sp(0), Bt('r2b'), Sp(0), Bt('r2c'),
                           Sp(gap),
                           Bt('r2a'), Sp(0), Bt('r2b'), Sp(0), Bt('r2c')]]

        img_fn = self._create_img_files(['test.png'])[0]
        Bi = gfx.BoxedImage
        bimgs_1st_col = [Bi(img_fn), Sp(0), Bi(img_fn), Sp(0), Bi(img_fn),
                         Sp(0, gap),
                         Bi(img_fn), Sp(0), Bi(img_fn), Sp(0), Bi(img_fn)]

        # logger.setLevel(logging.DEBUG)        
        gfx.adjust_hdr(row_hdr_btexts + [bimgs_1st_col],
                       use_height_for_last_line=True)

        self.assertEquals(row_hdr_btexts[0][0].get_box_width(), 1526)
        self.assertEquals(row_hdr_btexts[1][0].get_box_width(),
                          math.ceil((len(lg_txt) * 25. - gap) / 2))
        self.assertEquals(row_hdr_btexts[1][1].get_box_width(), gap)
        
        self.assertEquals(row_hdr_btexts[2][0].get_box_width(), 251)
        self.assertEquals(row_hdr_btexts[2][6].get_box_width(), 251)
        self.assertEquals(row_hdr_btexts[2][1].get_box_width(), 0)
        self.assertEquals(row_hdr_btexts[2][5].get_box_width(), gap)
        
        self.assertEquals(bimgs_1st_col[0].get_box_height(), 251)
        self.assertEquals(bimgs_1st_col[0].get_box_width(), 211)
        self.assertEquals(bimgs_1st_col[-1].get_box_height(), 251)
        self.assertEquals(bimgs_1st_col[-1].get_box_width(), 211)

        
    def test_adjust_table(self):
        
        Bt = gfx.BoxedText
        Sp = gfx.Spacer
        gap = 20
        short_txt = 'short col hdr'
        col_hdr_btexts = [[Bt(short_txt)],
                          [Bt('c1a'), Sp(gap), Bt('c1b')],
                          [Bt('c2a'), Sp(0), Bt('c2b'), Sp(0), Bt('c2c'),
                           Sp(gap),
                           Bt('c2a'), Sp(0), Bt('c2b'), Sp(0), Bt('c2c')]]
                          
        lg_txt = 'the very very very very very very very very very long row hdr'
        row_hdr_btexts = [[Bt(lg_txt)],
                          [Bt('r1a'), Sp(gap), Bt('r1b')],
                          [Bt('r2a'), Sp(0), Bt('r2b'), Sp(0), Bt('r2c'),
                           Sp(gap),
                           Bt('r2a'), Sp(0), Bt('r2b'), Sp(0), Bt('r2c')]]

        ifn = self._create_img_files(['test.png'])[0]

        Bi = gfx.BoxedImage
        bimgs = np.array([[Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)],
                          [Sp(0) for i in xrange(11)],
                          [Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)],
                          [Sp(0) for i in xrange(11)],
                          [Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)],
                          [Sp(0, gap) for i in xrange(11)],
                          [Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)],
                          [Sp(0) for i in xrange(11)],
                          [Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)],
                          [Sp(0) for i in xrange(11)],
                          [Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)]])

        assert len(bimgs[:,0]) == len(row_hdr_btexts[-1])
        assert len(bimgs[0,:]) == len(col_hdr_btexts[-1])
                                 
        #logger.setLevel(logging.DEBUG)
        gfx.adjust_table_sizes(row_hdr_btexts, col_hdr_btexts, bimgs)

        self.assertEquals(row_hdr_btexts[0][0].get_box_width(), 1526)
        self.assertEquals(row_hdr_btexts[1][0].get_box_width(), 753)
        self.assertEquals(bimgs[0][0].get_box_height(), 251)
        self.assertEquals(bimgs[0][0].get_box_width(), 211)
        self.assertEquals(bimgs[1][0].get_box_height(), 0)
        self.assertEquals(bimgs[1][0].get_box_width(), 211)
        self.assertEquals(bimgs[5][0].get_box_height(), gap)
        self.assertEquals(bimgs[5][0].get_box_width(), 211)        
        self.assertEquals(col_hdr_btexts[0][0].get_box_width(), 211*6 + gap)
        self.assertEquals(col_hdr_btexts[1][0].get_box_width(), 211*3)
        

    def test_arrange(self):
        Bt = gfx.BoxedText
        Sp = gfx.Spacer
        gap = 20
        short_txt = 'short col hdr'
        col_hdr_btexts = [[Bt(short_txt)],
                          [Bt('c1a'), Sp(gap), Bt('c1b')],
                          [Bt('c2a'), Sp(0), Bt('c2b'), Sp(0), Bt('c2c'),
                           Sp(gap),
                           Bt('c2a'), Sp(0), Bt('c2b'), Sp(0), Bt('c2c')]]
                          
        lg_txt = 'the very very very very very very very very very long row hdr'
        row_hdr_btexts = [[Bt(lg_txt)],
                          [Bt('r1a'), Sp(gap), Bt('r1b')],
                          [Bt('r2a'), Sp(0), Bt('r2b'), Sp(0), Bt('r2c'),
                           Sp(gap),
                           Bt('r2a'), Sp(0), Bt('r2b'), Sp(0), Bt('r2c')]]

        ifn = self._create_img_files(['test.png'])[0]

        Bi = gfx.BoxedImage
        bimgs = np.array([[Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)],
                          [Sp(0) for i in xrange(11)],
                          [Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)],
                          [Sp(0) for i in xrange(11)],
                          [Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)],
                          [Sp(0, gap) for i in xrange(11)],
                          [Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)],
                          [Sp(0) for i in xrange(11)],
                          [Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)],
                          [Sp(0) for i in xrange(11)],
                          [Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn), Sp(gap),
                           Bi(ifn), Sp(0), Bi(ifn), Sp(0), Bi(ifn)]])

        assert len(bimgs[:,0]) == len(row_hdr_btexts[-1])
        assert len(bimgs[0,:]) == len(col_hdr_btexts[-1])

        gfx.adjust_table_sizes(row_hdr_btexts, col_hdr_btexts, bimgs)

        gfx.arrange_table(row_hdr_btexts, col_hdr_btexts, bimgs)

        # Check column header
        self.assertTupleEqual(col_hdr_btexts[0][0].get_box_coords(), (0,0))
        self.assertTupleEqual(col_hdr_btexts[1][0].get_box_coords(), (0,40))
        self.assertTupleEqual(col_hdr_btexts[1][1].get_box_coords(), (633,40))
        self.assertTupleEqual(col_hdr_btexts[2][0].get_box_coords(), (0,80))
        self.assertTupleEqual(col_hdr_btexts[2][-1].get_box_coords(),
                              (5*211+gap,80))

        # Check row header
        self.assertTupleEqual(row_hdr_btexts[0][0].get_box_coords(), (0,0))
        self.assertTupleEqual(row_hdr_btexts[1][0].get_box_coords(), (0,40))
        self.assertTupleEqual(row_hdr_btexts[1][2].get_box_coords(),
                              (753+gap, 40))
        self.assertTupleEqual(row_hdr_btexts[2][0].get_box_coords(), (0,80))
        self.assertTupleEqual(row_hdr_btexts[2][-1].get_box_coords(),
                              (251*5 + gap, 80))

        # Check image grid
        self.assertTupleEqual(bimgs[0][0].get_box_coords(), (0,120))
        self.assertTupleEqual(bimgs[1][1].get_box_size(), (0,0))
        self.assertTupleEqual(bimgs[5][0].get_box_size(), (211,gap))
        self.assertTupleEqual(bimgs[5][1].get_box_size(), (0,gap))
        self.assertTupleEqual(bimgs[-1][-1].get_box_coords(),
                              (5*211+gap,120+251*5+gap))


    def test_boxed_img(self):
        img_fn = self._create_img_files(['test.png'])[0]
        self.assertTupleEqual(gfx.BoxedImage(img_fn, img_h=100).get_box_size(),
                              (round(211./239 * 100), 100))
                
class BoxedRectTest(CardoTest):

    def test_rect_size(self):
        br = cardo.graphics.BoxedRect(50, 75, 0, 0)

        br.set_box_width(100)
        br.set_box_height(50)

        self.assertTupleEqual(br.get_box_size(), (100, 75))

    def test_rect_coords(self):

        br = cardo.graphics.BoxedRect(50, 75, 0, 0, 
                                      halign=gfx.BoxedRect.HALIGN_CENTER,
                                      valign=gfx.BoxedRect.VALIGN_CENTER)

        br.set_box_width(100)
        br.set_box_height(50)

        
        self.assertTupleEqual(br.get_rect_coords(), (25, 0))

        
# class SizeTreeTest(CardoTest):


#     def test_adujst(self):

#         cardo.graphics.SizeTree.reset_sizes()
#         l3_e1 = cardo.graphics.SizeTree(2, 2) #->10
#         l3_e2 = cardo.graphics.SizeTree(2, 8) #->10
#         l3_e3 = cardo.graphics.SizeTree(2, 3) #->10
#         l2_e1 = cardo.graphics.SizeTree(1, 21, [l3_e1, l3_e2, l3_e3]) #->30
        
#         l3_e4 = cardo.graphics.SizeTree(2, 2) #->10
#         l3_e5 = cardo.graphics.SizeTree(2, 7) #->10
#         l3_e6 = cardo.graphics.SizeTree(2, 3) #->10
#         l2_e2 = cardo.graphics.SizeTree(1, 30, [l3_e4, l3_e5, l3_e6]) #->30

#         root = cardo.graphics.SizeTree(0, 20, [l2_e1, l2_e2]) #-> 60

#         root.adjust()

#         self.assertEquals(l3_e1.get_size(), 10)
#         self.assertEquals(l3_e2.get_size(), 10)
#         self.assertEquals(l2_e1.get_size(), 30)
        
#         self.assertEquals(l3_e4.get_size(), 10)
#         self.assertEquals(l2_e2.get_size(), 30)
        
#         self.assertEquals(root.get_size(), 60)
#         cardo.graphics.SizeTree.reset_sizes()

    
    # def __test_adujst_siblings(self):

    #     l3_e1 = cardo.graphics.SizeTree(2)
    #     l3_e2 = cardo.graphics.SizeTree(7)
    #     l3_e3 = cardo.graphics.SizeTree(3)

    #     l3_e4 = cardo.graphics.SizeTree(2)
    #     l3_e5 = cardo.graphics.SizeTree(8)
    #     l3_e6 = cardo.graphics.SizeTree(3)
        
    #     l2_e1 = cardo.graphics.SizeTree(21, [l3_e1, l3_e2, l3_e3])
    #     l2_e2 = cardo.graphics.SizeTree(30, [l3_e4, l3_e5, l3_e6])

    #     root = cardo.graphics.SizeTree(20, [l2_e1, l2_e2])

    #     root.adjust_to_siblings()

    #     self.assertEquals(l3_e1.get_size(), 8)
    #     self.assertEquals(l2_e1.get_size(), 30)
    #     self.assertEquals(root.get_size(), 20)

    # def __test_adujst_children(self):

    #     l3_e1 = cardo.graphics.SizeTree(2) #->7
    #     l3_e2 = cardo.graphics.SizeTree(8) #->8
    #     l3_e3 = cardo.graphics.SizeTree(3) #->7
    #     l2_e1 = cardo.graphics.SizeTree(21, [l3_e1, l3_e2, l3_e3]) #->22
        
    #     l3_e4 = cardo.graphics.SizeTree(2) #->10
    #     l3_e5 = cardo.graphics.SizeTree(7) #->10
    #     l3_e6 = cardo.graphics.SizeTree(3) #->10
    #     l2_e2 = cardo.graphics.SizeTree(30, [l3_e4, l3_e5, l3_e6]) #->30

    #     root = cardo.graphics.SizeTree(20, [l2_e1, l2_e2]) #-> 52

    #     root.adjust_to_children()

    #     self.assertEquals(l3_e1.get_size(), 7)
    #     self.assertEquals(l3_e2.get_size(), 8)
    #     self.assertEquals(l2_e1.get_size(), 22)
        
    #     self.assertEquals(l3_e4.get_size(), 10)
    #     self.assertEquals(l2_e2.get_size(), 30)
        
    #     self.assertEquals(root.get_size(), 52)    

def get_growth_profile_img_content():
    img_64 = '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'
    return base64.b64decode(img_64)
        
if __name__ == "__main__":
    unittest.main()

    
