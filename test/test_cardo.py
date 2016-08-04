import unittest
import tempfile
import shutil
import os.path as op
import os
from pprint import pformat
import re
import numpy as np
import base64
import logging

from .context import cardo
gfx = cardo.graphics

logger = logging.getLogger('cardo')

#cd ..;python -m unittest -v test.test_cardo

#TODO: check consistency of image grid


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

        #logger.setLevel(logging.DEBUG)
        
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

        
class ElementsTest(CardoTest):

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


    def test_boxed_img(self):
        img_fn = self._create_img_files(['test.png'])[0]
        self.assertTupleEqual(gfx.BoxedImage(img_fn, img_h=100).get_box_size(),
                              (round(211./239 * 100), 100))

    def test_img_inclusion(self):

        def get_iattr(svg, attr):
            pat = '<image.*%s="(.*?)" .*/>' % attr
            return re.match(pat, svg).group(1)

        img_fn = self._create_img_files(['level1/level2/test.png'])[0]

        img = gfx.BoxedImage(img_fn)
        isvg = img.to_svg().tostring()
        self.assertEquals(get_iattr(isvg, 'xlink:href'), img_fn)

        self.assertRaises(gfx.WrongSVGImgInclusion, img.to_svg,
                          inclusion='dafuq')

        isvg = img.to_svg(inclusion=gfx.BoxedImage.IMG_EXT_REL,
                          ref_path=op.join(self.tmp_dir, 'level1')).tostring()
        self.assertEquals(get_iattr(isvg, 'xlink:href'), 'level2/test.png')

        isvg = img.to_svg(inclusion=gfx.BoxedImage.IMG_EMBED).tostring()
        img_href = get_iattr(isvg, 'xlink:href')
        header =  'data:image/png;base64,'
        self.assertTrue(img_href.startswith(header))

        self.assertEquals(img_href[len(header):],
                          get_growth_profile_img_content64())
        
        
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


class OverlappingElementException(Exception):
    pass

class TableTest(CardoTest):

    def _check_no_overlap(self, elements):
        bot_coords = np.array([elem.get_box_bot_right_coords() \
                               for elem in elements])
        max_x, max_y = bot_coords.max(0)
        top_coords = np.array([elem.get_box_coords() for elem in elements])
        min_x, min_y = top_coords.max(0)
        self.assertGreaterEqual(min_x, 0)
        self.assertGreaterEqual(min_y, 0)
        
        grid_checker = np.zeros((max_x, max_y), dtype=int) - 1
        for ielem, element in enumerate(elements):
            logger.debug('check_overlap for %s', str(element))
            itop, jtop = element.get_box_coords()
            ibot, jbot = element.get_box_bot_right_coords()
            if itop != ibot and jtop != jbot: # do not check element with no area
                found_elements = np.unique(grid_checker[itop:ibot, jtop:jbot])
                logger.debug('matching elements -> %s', str(found_elements))
                if len(found_elements) > 1 or found_elements[0] != -1:
                    overlapping_elemts = [elements[i] for i in found_elements \
                                          if i != -1]
                    msg = 'Element %s overlaps with %s' \
                          %(str(element), str(overlapping_elemts))
                    raise OverlappingElementException(msg)
                grid_checker[itop:ibot, jtop:jbot] = ielem
    
    def test_check_overlap(self):

        r1 = gfx.BoxedRect(rwidth=211, rheight=40, box_x=0, box_y=80)
        r2 = gfx.BoxedRect(rwidth=211, rheight=239, box_x=0, box_y=80)

        self.assertRaises(OverlappingElementException,
                          self._check_no_overlap, [r1,r2], )


    def test_deoverlap_small(self):
        col_hdr_levels = [ ['a', 'b'] ]
        row_hdr_levels = [ ['h1', 'h2'] ]

        Bi = gfx.BoxedImage
        nb_elems_col = np.prod([len(lvl) for lvl in col_hdr_levels])
        nb_elems_row = np.prod([len(lvl) for lvl in row_hdr_levels])

        def get_ifn(ibfn):
            return self._create_img_files([ibfn])[0]

        imgs_bfn = [['h1_a.png', 'h1_b.png'],
                    ['h2_a.png', 'h2_b.png']]
        
        images = np.array([[Bi(get_ifn(imgs_bfn[i][j])) \
                            for j in xrange(nb_elems_col)] \
                            for i in xrange(nb_elems_row)], dtype=object)
        #logger.setLevel(logging.DEBUG)
        table = cardo.graphics.Table(row_hdr_levels, col_hdr_levels, images)
        table.deoverlap()

        relems = table.get_elements_via_row_header(walk_type='dfs')
        self.assertEquals(relems[0][0].text, 'h1')
        self.assertTrue(relems[0][1].img_fn.endswith('h1_a.png'))
        self.assertEquals(relems[1][0].text, 'h2')
        self.assertTrue(relems[1][1].img_fn.endswith('h2_a.png'))

        celems = table.get_elements_via_column_header(walk_type='dfs')
        self.assertEquals(celems[0][0].text, 'a')
        self.assertTrue(celems[0][1].img_fn.endswith('h1_a.png'))
        self.assertEquals(celems[1][0].text, 'b')
        self.assertTrue(celems[1][1].img_fn.endswith('h1_b.png'))
        
        row_hdr, col_hdr, imgs = table.get_table_parts()
        self._check_no_overlap(row_hdr)
        
        # test col hdr and imgs together as they are in the final orientation:
        self._check_no_overlap(col_hdr + imgs)
        
        
    def test_deoverlap_without_spacers(self):
        col_hdr_levels = [ ['a', 'b', 'c'],
                           ['1', '2'],
                           ['o', 'oo', 'ooo'] ]

        row_hdr_levels = [ ['h1', 'h2'],
                           ['r1', 'r2']]
        
        ifn = self._create_img_files(['test.png'])[0]
        Bi = gfx.BoxedImage
        nb_elems_col = np.prod([len(lvl) for lvl in col_hdr_levels])
        nb_elems_row = np.prod([len(lvl) for lvl in row_hdr_levels])
        images = np.array([[Bi(ifn) for j in xrange(nb_elems_col)] \
                            for i in xrange(nb_elems_row)], dtype=object)

        table = cardo.graphics.Table(row_hdr_levels, col_hdr_levels, images)
        table.deoverlap()

        row_hdr, col_hdr, imgs = table.get_table_parts()
        
        # test row_hdr separately because it's not in its final orientation
        # -> need to translate and rotate the whole group
        # -> but it's done when producing SVG using transforms
        # TODO: find a way to check that row hdr does not overlap with
        #       other elements in its final position
        self._check_no_overlap(row_hdr)
        
        # test col hdr and imgs together as they are in the final orientation:
        self._check_no_overlap(col_hdr + imgs)

    def test_deoverlap_with_spacers(self):
        col_hdr_levels = [ ['a', 'b', 'c'],
                           ['1', '2'],
                           ['o', 'oo', 'ooo'] ]

        row_hdr_levels = [ ['h1', 'h2'],
                           ['r1', 'r2']]
        
        ifn = self._create_img_files(['test.png'])[0]
        Bi = gfx.BoxedImage
        nb_elems_col = np.prod([len(lvl) for lvl in col_hdr_levels])
        nb_elems_row = np.prod([len(lvl) for lvl in row_hdr_levels])
        images = np.array([[Bi(ifn) for j in xrange(nb_elems_col)] \
                            for i in xrange(nb_elems_row)], dtype=object)

        #logger.setLevel(logging.DEBUG)
        table = cardo.graphics.Table(row_hdr_levels, col_hdr_levels, images)
        table.add_spacers()
        table.adjust_cell_sizes()
        table.deoverlap()

        row_hdr, col_hdr, imgs = table.get_table_parts()
        
        # test row_hdr separately because it's not in its final orientation
        # -> need to translate and rotate the whole group
        # -> but it's done when producing SVG using transforms
        # TODO: find a way to check that row hdr does not overlap with
        #       other elements in its final position
        self._check_no_overlap(row_hdr)
        
        # test col hdr and imgs together as they are in the final orientation:
        self._check_no_overlap(col_hdr + imgs)

        # check positions
        self.assertTupleEqual(col_hdr[0].get_box_coords(), (0,0))
        expected_next_x = 633 * 2 + gfx.Table.DEFAULT_COL_BGAP
        self.assertTupleEqual(col_hdr[1].get_box_coords(), (expected_next_x,0))
        expected_next_x += 2 * gfx.Table.DEFAULT_COL_BGAP
        self.assertTupleEqual(col_hdr[2].get_box_coords(), (expected_next_x, 0))

        self.assertTupleEqual(col_hdr[5].get_box_coords(),
                              (0, gfx.BoxedText.DEFAULT_FONT_H))
        self.assertTupleEqual(col_hdr[6].get_box_coords(),
                              (211 * 3, gfx.BoxedText.DEFAULT_FONT_H))

        self.assertTupleEqual(imgs[0].get_box_coords(),
                              (0, gfx.BoxedText.DEFAULT_FONT_H * 3))
        self.assertTupleEqual(imgs[1].get_box_coords(),
                              (211, gfx.BoxedText.DEFAULT_FONT_H * 3))
        self.assertTupleEqual(imgs[2].get_box_coords(),
                              (211, gfx.BoxedText.DEFAULT_FONT_H * 3))
        self.assertTupleEqual(imgs[6].get_box_coords(),
                              (211 * 3 + gfx.Table.DEFAULT_COL_BGAP,
                               gfx.BoxedText.DEFAULT_FONT_H * 3))

class GTreeTest(CardoTest):

    def test_from_hdr_and_images(self):

        hdr_levels = [ ['a', 'b', 'c'],
                       ['1', '2'],
                       ['o', 'oo', 'ooo'] ]

        ifn = self._create_img_files(['test.png'])[0]
        Bi = gfx.BoxedImage
        images = np.array([[Bi(ifn) for i in xrange(18)] for j in xrange(3)]).T
        root = cardo.graphics.GTree.from_hdr_and_images(hdr_levels, images)

        self.assertEquals(root.get_height(), len(hdr_levels) + images.shape[1])
        self.assertEquals(root.children[0].gfx_element.text, 'a')
        self.assertEquals(root.children[1].gfx_element.text, 'b')
        self.assertEquals(root.children[2].gfx_element.text, 'c')
        self.assertEquals(root.children[0].children[0].gfx_element.text, '1')
        img_1 = root.children[0].children[0].children[0].children[0].gfx_element
        self.assertIsInstance(img_1, cardo.graphics.BoxedImage)

    def test_spacing(self):
        
        hdr_levels = [ ['a', 'b', 'c'],
                       ['1', '2'],
                       ['o', 'oo', 'ooo'] ]

        ifn = self._create_img_files(['test.png'])[0]
        Bi = gfx.BoxedImage
        images = np.array([[Bi(ifn) for i in xrange(18)] for j in xrange(3)]).T
        root = cardo.graphics.GTree.from_hdr_and_images(hdr_levels, images)

        def create_spacer(gap, ref_sibling):
            return cardo.graphics.Spacer(width=gap,
                                         height=ref_sibling.get_box_height())
        root.add_spacers(10, create_spacer, 2)
        self.assertEquals(root.children[0].gfx_element.text, 'a')
        self.assertEquals(root.children[-1].gfx_element.text, 'c')
        self.assertIsInstance(root.children[1].gfx_element,
                              cardo.graphics.Spacer)
        self.assertTupleEqual(root.children[1].gfx_element.get_box_size(),
                              (20,40))
        child = root.children[1].children[0]
        self.assertTupleEqual(child.gfx_element.get_box_size(), (20,40))
        child = root.children[1].children[0].children[0]
        self.assertTupleEqual(child.gfx_element.get_box_size(), (20,40))
        child = root.children[1].children[0].children[0].children[0]
        self.assertTupleEqual(child.gfx_element.get_box_size(), (20,239))
        child = root.children[0].children[1]
        self.assertTupleEqual(child.gfx_element.get_box_size(), (10,40))

        for line in root.get_elements():
            self.assertEquals(len(set([e.get_box_height() for e in line])), 1)
        
    def test_resize(self):
        lg_txt = 'the very very very very very very very very very long row hdr'
        hdr_levels = [ [lg_txt],
                       ['r1a', 'r1b'],
                       ['r2a', 'r2b', 'r2c'] ]

        ifn = self._create_img_files(['test.png'])[0]
        Bi = gfx.BoxedImage
        images = np.array([[Bi(ifn) for i in xrange(6)] \
                           for j in xrange(3)]).T
        root = cardo.graphics.GTree.from_hdr_and_images(hdr_levels, images)

        #logger.setLevel(logging.DEBUG)
                
        def get_elem_size_col_hdr(elem):
            return elem.get_box_width()
        
        def set_elem_size_col_hdr(elem, s):
            return elem.set_box_width(s)

        for child in root.children:
            child.adjust_size(get_elem_size_col_hdr, set_elem_size_col_hdr)
            
        subnode = root.children[0]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (1525,40))
        subnode = subnode.children[-1]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (762,40))
        subnode = subnode.children[-1]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (254,40))
        subnode = subnode.children[-1]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (254, 239))

    def test_resize_short_hdr(self):
        hdr_levels = [ ['top'],
                       ['r1a', '-'*27],
                       ['r2a', 'r2b', 'r2c'] ]

        ifn = self._create_img_files(['test.png'])[0]
        Bi = gfx.BoxedImage
        images = np.array([[Bi(ifn) for i in xrange(6)] for j in xrange(3)]).T
        root = cardo.graphics.GTree.from_hdr_and_images(hdr_levels, images)

        #logger.setLevel(logging.DEBUG)
                
        def get_elem_size_col_hdr(elem):
            return elem.get_box_width()
        
        def set_elem_size_col_hdr(elem, s):
            return elem.set_box_width(s)

        for child in root.children:
            child.adjust_size(get_elem_size_col_hdr, set_elem_size_col_hdr)
            
        subnode = root.children[0]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (1349,40))
        subnode = subnode.children[-1]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (675,40))
        subnode2 = root.children[0].children[0]
        self.assertTupleEqual(subnode2.gfx_element.get_box_size(), (674,40))
        subnode = subnode.children[-1]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (225,40))
        subnode = subnode.children[-1]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (225, 239))


    def test_resize_with_gaps(self):
        hdr_levels = [ ['top'],
                       ['r1a', '-'*27],
                       ['r2a', 'r2b', 'r2c'] ]

        ifn = self._create_img_files(['test.png'])[0]
        Bi = gfx.BoxedImage
        images = np.array([[Bi(ifn) for i in xrange(6)] for j in xrange(3)]).T
        root = cardo.graphics.GTree.from_hdr_and_images(hdr_levels, images)

        base_gap = 10
        def create_spacer(gap, ref_sibling):
            return cardo.graphics.Spacer(gap, ref_sibling.get_box_height())
        root.add_spacers(base_gap, create_spacer,
                         nb_levels_to_space=len(hdr_levels))
        
        #logger.setLevel(logging.DEBUG)
                
        def get_elem_size_col_hdr(elem):
            return elem.get_box_width()
        
        def set_elem_size_col_hdr(elem, s):
            return elem.set_box_width(s)

        for child in root.children:
            child.adjust_size(get_elem_size_col_hdr, set_elem_size_col_hdr)
            
        subnode = root.children[0]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (1369,40))
        subchildren = subnode.children
        self.assertTupleEqual(subchildren[0].gfx_element.get_box_size(), (674,40))
        self.assertTupleEqual(subchildren[1].gfx_element.get_box_size(), (20,40))
        self.assertTupleEqual(subchildren[1].children[0].gfx_element.get_box_size(),
                              (20,40))
        self.assertTupleEqual(subchildren[2].gfx_element.get_box_size(), (675,40))
        subchildren = subnode.children[-1].children
        self.assertTupleEqual(subchildren[0].gfx_element.get_box_size(), (218,40))
        self.assertTupleEqual(subchildren[1].gfx_element.get_box_size(), (10,40))
        subnode = subchildren[0].children[-1]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (218, 239))
        
        
    def test_get_elements(self):
        hdr_levels = [ ['the top hdr'],
                       ['r1a', 'r1b'],
                       ['r2a', 'r2b', 'r2c'] ]

        def get_ifn(irow, icol):
            return self._create_img_files(['%d_%d.png' %(irow, icol)])[0]

        Bi = gfx.BoxedImage
        images = np.array([[Bi(get_ifn(i,j)) for j in xrange(6)] \
                           for i in xrange(3)]).T
        
        
        root = cardo.graphics.GTree.from_hdr_and_images(hdr_levels, images)

        # bfs, leveled
        elements = root.get_elements()

        self.assertEquals(elements[0][0].text, 'the top hdr')
        self.assertEquals(elements[1][0].text, 'r1a')
        self.assertEquals(elements[1][1].text, 'r1b')
        self.assertEquals(elements[2][0].text, 'r2a')
        self.assertEquals(elements[2][3].text, 'r2a')
        self.assertEquals(elements[2][3].text, 'r2a')
        self.assertTrue(elements[3][0].img_fn.endswith('0_0.png'))
        self.assertTrue(elements[3][-1].img_fn.endswith('0_5.png'))
        self.assertTrue(elements[4][0].img_fn.endswith('1_0.png'))
        self.assertTrue(elements[4][-1].img_fn.endswith('1_5.png'))
        self.assertTrue(elements[-1][-1].img_fn.endswith('2_5.png'))

        # bfs, not leveled
        elements = root.get_elements(leveled=False)
        self.assertEquals(elements[0].text, 'the top hdr')
        self.assertEquals(elements[1].text, 'r1a')
        self.assertEquals(elements[2].text, 'r1b')
        self.assertEquals(elements[3].text, 'r2a')
        self.assertEquals(elements[6].text, 'r2a')
        self.assertTrue(elements[9].img_fn.endswith('0_0.png'))
        self.assertTrue(elements[14].img_fn.endswith('0_5.png'))
        self.assertTrue(elements[15].img_fn.endswith('1_0.png'))
        self.assertTrue(elements[20].img_fn.endswith('1_5.png'))
        self.assertTrue(elements[-1].img_fn.endswith('2_5.png'))
        
        # dfs, leveled
        #logger.setLevel(logging.DEBUG)
        elements = root.get_elements(walk_type='dfs')
        self.assertEquals(elements[0][0].text, 'the top hdr')
        self.assertEquals(elements[0][1].text, 'r1a')
        self.assertEquals(elements[0][2].text, 'r2a')
        self.assertTrue(elements[0][3].img_fn.endswith('0_0.png'))
        self.assertTrue(elements[0][4].img_fn.endswith('1_0.png'))
        self.assertEquals(elements[1][1].text, 'r1a')
        self.assertEquals(elements[1][2].text, 'r2b')
        self.assertTrue(elements[1][3].img_fn.endswith('0_1.png'))
        self.assertTrue(elements[1][4].img_fn.endswith('1_1.png'))
        self.assertTrue(elements[1][5].img_fn.endswith('2_1.png'))
        self.assertTrue(elements[2][4].img_fn.endswith('1_2.png'))
        self.assertTrue(elements[5][4].img_fn.endswith('1_5.png'))
        self.assertTrue(elements[-1][-1].img_fn.endswith('2_5.png'))

    def test_get_element_partial(self):

        hdr_levels = [ ['the top hdr'],
                       ['r1a', 'r1b'],
                       ['r2a', 'r2b', 'r2c'] ]

        def get_ifn(irow, icol):
            return self._create_img_files(['%d_%d.png' %(irow, icol)])[0]

        Bi = gfx.BoxedImage
        images = np.array([[Bi(get_ifn(i,j)) for j in xrange(6)] \
                           for i in xrange(3)]).T

        root = cardo.graphics.GTree.from_hdr_and_images(hdr_levels, images)

        # dfs, skip hdr
        elements = root.get_elements(walk_type='dfs', start_depth=3)
        self.assertTrue(elements[0][0].img_fn.endswith('0_0.png'))
        self.assertTrue(elements[0][1].img_fn.endswith('1_0.png'))
        self.assertTrue(elements[1][0].img_fn.endswith('0_1.png'))
        self.assertTrue(elements[1][1].img_fn.endswith('1_1.png'))
        self.assertTrue(elements[1][2].img_fn.endswith('2_1.png'))
        self.assertTrue(elements[2][1].img_fn.endswith('1_2.png'))
        self.assertTrue(elements[5][1].img_fn.endswith('1_5.png'))
        self.assertTrue(elements[-1][-1].img_fn.endswith('2_5.png'))

        # bfs, skip hdr
        elements = root.get_elements(walk_type='bfs', start_depth=3)
        self.assertTrue(elements[0][0].img_fn.endswith('0_0.png'))
        self.assertTrue(elements[0][-1].img_fn.endswith('0_5.png'))
        self.assertTrue(elements[1][0].img_fn.endswith('1_0.png'))
        self.assertTrue(elements[1][-1].img_fn.endswith('1_5.png'))
        self.assertTrue(elements[-1][-1].img_fn.endswith('2_5.png'))

        
        
    def test_resize_with_spacers(self):
        lg_txt = 'the very very very very very very very very very long row hdr'
        hdr_levels = [ [lg_txt],
                       ['r1a', 'r1b'],
                       ['r2a', 'r2b', 'r2c'] ]

        ifn = self._create_img_files(['test.png'])[0]
        Bi = gfx.BoxedImage
        images = np.array([[Bi(ifn) for i in xrange(6)] for j in xrange(3)]).T

        #print 'create gtree from hdr and images ...'
        root = cardo.graphics.GTree.from_hdr_and_images(hdr_levels, images)

        def get_elem_size_col_hdr(elem):
            return elem.get_box_width()
        
        def set_elem_size_col_hdr(elem, s):
            return elem.set_box_width(s)

        def create_spacer(gap, ref_sibling):
            return cardo.graphics.Spacer(width=gap,
                                         height=ref_sibling.get_box_height())
        #print 'add spacers ...'        
        root.add_spacers(10, create_spacer, 2)

        #print 'adjust sizes ...'
        #logger.setLevel(logging.DEBUG)
        for child in root.children:
            child.adjust_size(get_elem_size_col_hdr, set_elem_size_col_hdr)
            
        subnode = root.children[0]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (1525,40))
        subnode = subnode.children[-1]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (757,40))
        subnode = subnode.children[-1]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (252,40))
        subnode = subnode.children[-1]
        self.assertTupleEqual(subnode.gfx_element.get_box_size(), (252, 239))
        


def get_growth_profile_img_content64():
    img_64 = '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'
    return img_64
                          
def get_growth_profile_img_content():
    return base64.b64decode(get_growth_profile_img_content64())
        
if __name__ == "__main__":
    unittest.main()

    
